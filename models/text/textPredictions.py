import os
import sys
import numpy as np
from utils import *
import pprint as pp
import time

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
from aip import AipNlp


class TextPredictions:
    def __init__(self, _config, verbose=False):
        _config = get_config(_config)
        config = _config['modalities']['text']['model_info']['default']
        self._config = config

        """ 你的 APPID AK SK """
        APP_ID = '17722042'
        API_KEY = 'XregIKAqLTteohz60Nk5rGYA'
        SECRET_KEY = 'jpudC8Pwjcjdz8fCHoaluUfGYSTCkGrj'

        self.client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        self.labels = list(label_mapping['baidu_text'].values())

    def predict_sample(self, data, return_all=False):
        if isinstance(data, list):
            data = data[0]
        """ 调用情感倾向分析 """
        try:
            res = self.client.sentimentClassify(data)
        except:
            print(data)
            res={'log_id': 110, 'text': '发生错误', 'items': [{'positive_prob': 0.5, 'confidence': 0.0, 'negative_prob': 0.5, 'sentiment': 0}]}

        print(res)
        item_res = res.get('items', [{}])[0]
        return_res = [item_res.get(item, 0)
                      for item in list(label_mapping['baidu_text'].values())]
        label = np.argmax(return_res)
        if return_all:
            return label, return_res, self.labels
        else:
            return label, return_res, self.labels

    def predict_sample_conv(self, data, return_all=False):
        if isinstance(data, list):
            data = data[0]
        """ 调用情感倾向分析 """
        res = self.client.emotion(data, options={'scene': 'talk'})
        item_res = res['items']
        pp.pprint(item_res)

        return_res = {item.get('label', 0): item.get('prob', 0) for item in item_res}
        return_res = [return_res[i] for i in ['neutral', 'pessimistic', 'optimistic']]
        replies_res = {item.get('label', 0): item.get('replies', []) for item in item_res}
        replies_res = [replies_res[i] for i in ['neutral', 'pessimistic', 'optimistic']]
        label_res = {item.get('label', 0): item.get('label', '') for item in item_res}
        label_res = [label_res[i] for i in ['neutral', 'pessimistic', 'optimistic']]
        label_idx = np.argmax(return_res)
        if return_all:
            return label_idx, return_res, label_res
        else:
            return {
                'label_idx': label_idx,
                'prob': return_res,
                'label': label_res,
                'reply': replies_res
            }

    def predict(self, data, return_all=False):
        # read image
        res = []
        for i, data_list in enumerate(data):
            source = data_list
            label, emotion_predictions, labels = self.predict_sample(source, return_all)
            res.append([label, self.labels[label], emotion_predictions])
            time.sleep(0.7)
        if return_all:
            return res, labels
        return res


emotion_map = {
    'text_emotion': {
        'happy': 'positive',
        'angry': 'negative',
        'sad': 'negative',
        'disgust': 'negative'
    },
    'conv_emotion': {
        'happy': 'optimistic',
        'angry': 'pessimistic',
        'sad': 'pessimistic',
        'disgust': 'pessimistic'
    }
}
label_2_index = {
    'text_emotion': {
        'positive': 0,
        'negative': 1
    },
    'conv_emotion': {
        'neutral': 0,
        'pessimistic': 1,
        'optimistic': 2,
        'positive': 2,
        'negative': 1
    }
}

text_type = 'text_emotion'
def get_test_data(file, text_type = 'text_emotion'):
    with open(file) as f:
        lines = f.readlines()
        lines = [l.strip().split('&') for l in lines if '&' in l]
        lines = [[l[0], l[1].split('（')[0], emotion_map[text_type][l[1].split('（')[0]]] for l in lines]
    return lines


if __name__ == '__main__':
    config = get_config("{}/configs/basic.json".format(root_path))
    face_predictor = TextPredictions(config)
    test_sample = False
    if test_sample:
        text = '好难啊'
        res = face_predictor.predict_sample_conv(text, return_all=True)
        print(res)
    else:
        lines = get_test_data('data/test_conversation.txt',text_type)
        print(len(lines))
        # print(lines)
        result_samples = [{}, {}]
        for i, l in enumerate(lines):
            print(i)
            print(l[0])
            if text_type in ['conv_emotion']:
                res = face_predictor.predict_sample_conv(l[0])
                print(res)
                emotion = res['prob']
                emotion_label = np.argmax(emotion)
                labels = res['label']
                confidence = None
                reply = res['reply']
            else:
                res = face_predictor.predict_sample(l[0])
                print(res)
                emotion = res[1][:2]
                emotion_label = np.argmax(emotion)
                labels = res[2]
                confidence = res[1][2]
                reply = None
                logits = res[1]
            label_gen = emotion_label
            label_gt = label_2_index[text_type][l[2]]
            print('label', label_gen, label_gt)
            result_samples[label_gen == label_gt][l[0]] = {
                'label': [l[1], l[2]],
                'predict': labels[emotion_label],
                'confidence': confidence,
                'logits': emotion,
                'reply': reply
            }
            import time

            time.sleep(1)
        print()
        right_samples = result_samples[1]
        bad_samples = result_samples[0]
        print('result')
        print(len(right_samples))
        pp.pprint(right_samples)
        print(len(bad_samples))
        pp.pprint(bad_samples)
    with open('bad_res_baidu.txt','w') as f:
        f.write('\n'.join(list(bad_samples.keys())))
