import os
import sys
import torchvision.models as models
import torch
import cv2
import argparse
import os
import time
import json
import sys
import dlib
import pandas as pd
import numpy as np
import imutils

root_path = os.path.abspath(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        os.path.pardir)
)
if root_path not in sys.path:
    sys.path.append(root_path)

from utils import *

import numpy as np
import pprint as pp


# 该类实现如下功能：
# 1 多帧视频的时序信息融合，包括了引入衰减因子的方法和概率加和的方法
# 1.1 衰减因子：当前帧的预测结果会将之前帧的结果以一定比例进行修改，希望时序之间具有连续性，而不是跳转过大
# 1.2 概率加权：说一段话的时候，因为口型的改变，会导致判断不同的表情，比如说惊讶，所以希望提取出一整段视频中概率最高的表情出来
# 2 模态之间的信息融合
# 2.1 图片和音频都可以简单融合，但是文本需要进行转化，目前文本是两种情绪，以一定比例增加在图片和声音的正向和负面的情绪之中
# 2.2 Text置信度高的时候：Text》Image》Audio
# 2.3 Text置信度低的时候：Image》Text》Audio
# 2.4 Text的置信度需要more than一定的阈值才会加入ensemble
# 3 未来可能的优化：采用feature层面的融合来重新预测，目前的问题是没有合适的对齐的数据集，并且可能数据中很大可能是以单模态的形式存在的
# 4 可视化：通过雷达图的形式展现整体的情绪的预测
# input: 'data'中 {modal: {
# 'result':[[result, result_label, logits],[result, result_label, logits],...]
# 'labels': 标签的类别
# }}
class EnsembleUtil:
    def __init__(self, _config, verbose=False):
        self._verbose = False
        self._config = _config
        self.labels = basic_labels
        self.label2idx = {label: i for i, label in enumerate(self.labels)}
        self.rs = {
            "face": 0.2
        }

    def ensemble(self, output, modalities_dict=None):
        total_logits = np.zeros(len(self.labels))  # 最终的logits
        image_logits = []
        image_logits_sum = np.zeros(len(self.labels))  # image's logits summarization
        data = output['data']
        info = output['info']
        used_modality_list = list(set(info['used_modality_list']) - {'video'})
        # 对于所有标签进行对齐
        modal_label = info['modal_labels']
        modal_label_map = {modal: {l: word2basicemotion.get(words2word.get(l, l), None)
                                   for l in lables} for modal, lables in modal_label.items()}
        modal_label_index_map = {modal: {l: i
                                         for i, l in enumerate(lables)} for modal, lables in modal_label.items()}
        # 遍历所有index
        for index, value in data.items():
            modalities = value['modalities']
            # 对所有logits进行处理，映射成basic的格式
            for modal in used_modality_list:
                if modal in modalities.keys():
                    modal_res = modalities[modal]['result']
                    if isinstance(modal_res, list):
                        if modal in ['face']:
                            try:
                                modal_res_withoutfacerec = np.array(
                                    [m[2][0][0] for m in modal_res])  # 第一个0是拆解人，第二个0是拆解logits出来
                            except Exception as e:
                                print(e)
                        else:
                            modal_res_withoutfacerec = np.array(
                                [m[2] for m in modal_res])  # 第一个0是拆解人，第二个0是拆解logits出来
                        basic_logits_array = self._get_basic_logits_array(modal, modal_label_index_map,
                                                                          modal_label_map,
                                                                          modal_res_withoutfacerec)
                        modalities[modal + '_basic'] = {'result': basic_logits_array}
                        # if modal not in ['face']:
                        #     modalities[modal + '_basic_label'] = basic_labels[np.argmax(basic_logits_array)]
            # 对图片中的时序信息进行融合
            # 平滑因子，以r进行衰减
            is_max_pooling = self._config['ensemble']['method'].get('global_max_pooling',False) or self._config['ensemble']['method'].get('exception_max_pooling',False)
            for modal in ['face', 'pose']:
                if not modal in value['modalities'].keys():
                    continue
                # 得到image的所有预测的logits
                org_image_logits = value['modalities'][modal]['result']  # 得到一个list [resut，result-label，logits]
                # 获取衰减因子
                r = self.rs[modal]
                # 保存原本的求和结果
                sum_image_logits = np.zeros(len(self.labels))
                # 保存衰减之后的求和结果
                sum_image_logits_smooth = np.zeros(len(self.labels))
                # 获取基本emotion的logits array
                basic_image_logits_array = modalities[modal + '_basic']['result']
                # 初始化基本emotion smooth之后的logits array
                basic_image_logits_smooth_array = np.zeros(basic_image_logits_array.shape)
                max_one_logits = []
                for i in range(len(basic_image_logits_array)):
                    one_image_logits = basic_image_logits_array[i]
                    # record max logit to get max pooling
                    max_one_logit = np.max(basic_image_logits_array[i])
                    max_one_logit_index = np.argmax(basic_image_logits_array[i])
                    if self._config['ensemble']['method'].get('global_max_pooling', False):
                        max_one_logits.append(max_one_logit)
                    elif self._config['ensemble']['method'].get('exception_max_pooling', False):
                        if basic_labels[max_one_logit_index] in ['calm']:
                            max_one_logits.append(0)
                        else:
                            max_one_logits.append(max_one_logit)

                    sum_image_logits += one_image_logits
                    if i == 0:
                        curr_image_logits = one_image_logits
                        basic_image_logits_smooth_array[i] = curr_image_logits
                        sum_image_logits_smooth += curr_image_logits
                        continue
                    curr_image_logits = r * basic_image_logits_smooth_array[i - 1] + one_image_logits
                    curr_image_logits = softmax(curr_image_logits)
                    basic_image_logits_smooth_array[i] = curr_image_logits
                    sum_image_logits_smooth += curr_image_logits
                modalities[modal + '_basic_smooth'] = {'result': basic_image_logits_smooth_array}
                modalities[modal + '_basic_sum'] = {'result': [softmax(list(sum_image_logits))]}
                # modalities[modal + '_basic_sum'] = {'result': [list(sum_image_logits / len(basic_image_logits_array))]}
                modalities[modal + '_basic_smooth_sum'] = {'result': [softmax(list(sum_image_logits_smooth))]}

                if is_max_pooling:
                    max_index = np.argmax(max_one_logits)
                    modalities[modal + '_gloabl_max_pooling'] = {'result': [basic_image_logits_array[max_index]]}
                    if self._verbose:
                        print('{} max index: {}'.format(modal, max_index))
                if self._config['ensemble']['method'].get('exception_voting',False):
                    sum_image_logits[basic_labels.index('calm')] = 0
                    modalities[modal + '_exception_voting'] = {'result': [softmax(list(sum_image_logits))]}

            # 对模态之间的结果进行融合
            replace_map = {
                'face': 'face_basic_smooth_sum',
                'audio': 'audio_basic',
                'text': 'text_basic',
                'ensemble': 'ensemble_basic',
                'weight': 'weight_basic'
            }
            if self._config and not self._config['demo']['ensemble_smooth']:
                replace_map['face'] = 'face_basic_sum'
            if is_max_pooling:
                replace_map['face'] = 'face_gloabl_max_pooling'
            elif self._config['ensemble']['method'].get('exception_voting',False):
                replace_map['face'] = 'face_exception_voting'
            self.replace_map_final = replace_map
            ensembel_param = {
                'face': {'weight': 0.7},
                'audio': {'weight': 0},#0.4
                'text': {'weight': 0.3, 'threshold': 0.5},
                'pose': {'weight': 0},
            }
            extra_weight = {
                'face': {'weight': 2},
                'audio': {'weight': 0},#0.5
                'text': {'weight': 0.5},
                'pose': {'weight': 0},
            }
            basic_sum_logits = np.zeros((1, len(self.labels)), dtype=np.float)
            modalities['weight_basic'] = {'result': {}}
            for modal in used_modality_list:
                modal_final = replace_map.get(modal, modal)
                curr_modal_res = modalities[modal_final]['result']
                logits_idx = np.argmax(modalities[modal_final]['result'][0])
                max_logit = modalities[modal_final]['result'][0][logits_idx]
                ensembel_param[modal]['weight'] = max_logit
                modalities[modal_final + '_label'] = basic_labels[logits_idx]
                weight = 0
                if modalities_dict and modal in ['audio', 'text']:
                    text = modalities_dict['text']['file_dict'][index][0]
                    modalities['weight_basic']['result'][modal] = weight
                    if len(text) == 0:
                        continue
                # 如果不是标准格式这里可以做一个替换
                if modal in ['text']:
                    threshold = ensembel_param[modal].get('threshold', 0)
                    confidence = modalities[modal]['result'][0][2][-1]
                    if confidence > threshold:
                        weight = ensembel_param[modal].get('weight', 0.0)
                        # weight = confidence
                else:
                    weight = ensembel_param[modal].get('weight', 0.0)
                weight *= extra_weight[modal].get('weight', 1)
                basic_sum_logits += curr_modal_res[0] * weight
                modalities['weight_basic']['result'][modal] = weight
            basic_sum_logits = softmax(basic_sum_logits)
            modalities['ensemble_basic'] = {'result': basic_sum_logits}
            modalities['ensemble_basic_label'] = basic_labels[np.argmax(basic_sum_logits)]
            if self._verbose:
                print('basic_sum_logits')
                pp.pprint(basic_sum_logits)
                for modal in used_modality_list:
                    print(modal)
                    print(modalities[replace_map.get(modal, modal)]['result'])

    def _get_basic_logits_array(self, modal, modal_label_index_map, modal_label_map,
                                org_image_logits_array):
        new_shape = list(org_image_logits_array.shape)
        new_shape[-1] = len(self.labels)
        basic_image_logits_array = np.zeros(new_shape)
        # 获取当前modality从emotion到basic emotion的映射
        modal_emotion2basic = modal_label_map[modal]
        # 获取当前modality到index的映射
        modal_emotion2index = modal_label_index_map[modal]
        for emotion, map_list in modal_emotion2basic.items():
            if map_list is None:
                continue
            for map_res in map_list:
                # 需要把emotion的概率依次加到map res中单词的索引上去
                basic_image_logits_array[:, self.label2idx[map_res]] = org_image_logits_array[:,
                                                                       modal_emotion2index[emotion]]
        return basic_image_logits_array


if __name__ == "__main__":
    _modal_data = {'info': {
        'modal_labels': {'audio': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
                         'text': ['positive_prob', 'negative_prob', 'confidence'],
                         'face': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']},
        'index_modal': 'audio', 'index_list': ['01-01-01-01-02-01-02'],
        'used_modality_list': ['audio', 'video', 'face', 'text']},
        'data': {'01-01-01-01-02-01-02':
                     {'label': 'calm', 'modalities':
                         {'audio':
                              {'result': [[0, 'neutral',
                                           np.array([
                                               9.9874014e-01,
                                               3.4479951e-04,
                                               6.4196465e-05,
                                               4.3902775e-05,
                                               1.2139147e-06,
                                               1.2910414e-08,
                                               1.0686301e-08,
                                               8.0578518e-04])]]},
                          'video': {}, 'face': {'result': [
                             [0, 'neutral', np.array([5.5552918e-01, 7.4197143e-02, 2.0882897e-03, 2.0510295e-06,
                                                      3.3034059e-01, 2.8871984e-04, 2.1300874e-09, 3.7553947e-02])],
                             [0, 'neutral', np.array([6.2426507e-01, 7.2898204e-03, 8.4170076e-04, 5.5197920e-06,
                                                      3.5765940e-01, 3.9508665e-04, 1.9225550e-09, 9.5434375e-03])]]},
                          'text': {'result': [
                              [0, 'positive_prob',
                               [0, 0, 0]]]}}}}}
    # pp.pprint(_modal_data)
    ensemble_util = EnsembleUtil(None)
    ensemble_util.ensemble(_modal_data)
