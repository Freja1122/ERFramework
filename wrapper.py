import os
import pprint
from pandas import read_csv
import numpy as np
import shutil
import cv2
from shutil import copyfile
import time
import threading
import speech_recognition as sr
import logging
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dlib
from imutils import face_utils
from collections import Counter
# from keras.models import load_model
import face_recognition
from statistics import mode
import pickle
from utils import *
from models import *
from utils.face.datasets import get_labels
from utils.face.inference import detect_faces
from utils.face.inference import draw_text
from utils.face.inference import draw_bounding_box
from utils.face.inference import apply_offsets
from utils.face.inference import load_detection_model
from utils.face.preprocessor import preprocess_input

root_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
emotion_offsets = (20, 40)


class Wrapper:
    def __init__(self, config, interactive=False, verbose=False):
        self._verbose = verbose
        self._interactive = interactive
        self._output_prefix = '[Wrapper output]'
        print(self._output_prefix, "initializing wrapper")
        self._config = get_config(config)
        self._demo_display = self._config['demo']['display']
        self._demo_display_static = self._config['demo']['display_static']
        self._face_smooth = self._config['demo']['smooth']
        self._face_smooth_r = self._config['demo']['smooth_r']
        self._last_face_logits = None
        self._multi_person = self._config['demo']["multi_person"]
        self._target_name = self._config['demo']['target_name']
        modal_list = ['face', 'audio', 'text', 'note']
        for i in range(len(modal_list) - 1):
            self._demo_display[modal_list[i + 1]]['x'] = self._demo_display[modal_list[i]]['x']
            self._demo_display[modal_list[i + 1]]['y'] = self._demo_display[modal_list[i]]['y'] + \
                                                         self._demo_display[modal_list[i]]['h'] + \
                                                         self._demo_display['blank']
            self._demo_display_static[modal_list[i + 1]]['x'] = self._demo_display_static[modal_list[i]]['x']
            self._demo_display_static[modal_list[i + 1]]['y'] = self._demo_display_static[modal_list[i]]['y'] + \
                                                                self._demo_display_static[modal_list[i]]['h'] + \
                                                                self._demo_display_static['blank']
        modal_list = ['ensemble', 'weight']
        for i in range(len(modal_list) - 1):
            self._demo_display_static[modal_list[i + 1]]['x'] = self._demo_display_static[modal_list[i]]['x']
            self._demo_display_static[modal_list[i + 1]]['y'] = self._demo_display_static[modal_list[i]]['y'] + \
                                                                self._demo_display_static[modal_list[i]]['h'] + \
                                                                self._demo_display_static['blank']

        # self._demo_display['note']['y'] += 60
        self._modality_config = self._config['modalities']
        self._all_modality_list = [modal for modal in list(self._modality_config.keys())]
        self._get_data_folder(interactive)
        self.extractors = ModalityExtractor(self._config, interactive, verbose)
        self.ensemble_util = EnsembleUtil(self._config, self._verbose)
        # 初始化predictor
        if not self._config['caches']['run']:
            self.predictor = {
                'audio': {"default": None},
                'face': {"default": None,
                         "face_and_emotion": None},
                'text': {"default": None}
            }
        else:
            self.predictor = {
                'audio': {"default": AudioPredictions(self._config)},
                'face': {"default": FacePredictions(self._config),
                         "face_and_emotion": FacePredictions_faceandemotion(self._config)},
                'text': {"default": TextPredictions(self._config)}
            }
        self.predictor_select = {
            'audio': self._config['modalities']['audio']['model_info']['model_name'],
            'face': self._config['modalities']['face']['model_info']['model_name'],
            'text': self._config['modalities']['text']['model_info']['model_name']
        }
        if not interactive:
            # 根据modality的state信息提取文件名
            # 如果是extract，就从视频中提取，获取提取后文件的名称
            # 如果是True，就从文件中读取文件名
            # 如果是False，代表不使用这个模态
            self._extract_modality()
            if len(list(self._modalities_dict.keys())) == 0:
                raise ValueError('There is no modality')
            # 根据提取的文件名称，构建sample的数据
            # 'info'中包括了index的模态信息，data中包括了每一个sample的不同模态
            self._construct_sample_data()
            if self._config['caches']['modal_data']['state']:
                load_filename = os.path.join(root_path, self._config['caches']['modal_data']['filename'])
                with open(load_filename, 'rb') as handle:
                    self._modal_data = pickle.load(handle)
                    print('load modal data from {}'.format(load_filename))
            else:
                self._predict_all_modality()
                with open(self._config['caches']['modal_data']['filename'], 'wb') as handle:
                    pickle.dump(self._modal_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if self._config['modalities']['video']['state'] == True:
                self.ensemble_util.ensemble(self._modal_data, self._modalities_dict)
            # if self._verbose:
            #     self._print_extract_result()
            if self._config['modalities']['video']['state'] == True:
                self.video_statistic()

        if interactive:
            print(self._output_prefix, 'interactive mode')
            # 从camera中获取图片和声音，送到模型进行预测
            self._demo()

    def video_statistic(self):
        modal_data_res = self._modal_data['data']
        all_items = list(modal_data_res.items())
        data_len = len(all_items)
        real_bad_len = 0
        bad_len = 0
        ignore_len = 0
        output_bad_dir = self.mkdir_dir('video_bad_output')
        output_good_dir = self.mkdir_dir('video_good_output')
        mix_labels = ['sad', 'angry']
        for k, v in all_items:
            gt_label = v['label'][0]
            if gt_label not in basic_labels:
                ignore_len += 1
                continue
            gen_label = v['modalities']['ensemble_basic_label']
            if gt_label != gen_label:
                if not(gt_label in mix_labels and gen_label in mix_labels):
                    real_bad_len += 1
                bad_len += 1
                self.write_output_video(gen_label, k, output_bad_dir, v)
            else:
                self.write_output_video(gen_label, k, output_good_dir, v)

        predict_data_len = data_len - ignore_len
        print('accuracy:', float(predict_data_len - bad_len) / float(predict_data_len))
        print('except mix emotion accuracy:', float(predict_data_len - real_bad_len) / float(predict_data_len))
        print('data_len:', data_len)
        print('predict_data_len:', predict_data_len)
        print('bad_len:', bad_len)
        print('real_bad_len:', real_bad_len)
        print('ignore_len:', ignore_len)
        print('labels_count',self.labels_count)

    def write_output_video(self, gen_label, k, output_dir, v):
        # write file
        output_file_name = k + '-*ensemble*' + gen_label
        modal_res = {}
        print('bad result: ', k)
        for modal in self._modality_list:
            if modal in ['video']:
                continue
            modal_res[modal] = v['modalities'][
                self.ensemble_util.replace_map_final[modal] + '_label']
            output_file_name += '-*{}*{}'.format(modal, modal_res[modal])
        output_file_name += '-*{}*{}'.format('text', self._modalities_dict['text']['file_dict'][k][0])
        output_file_name_org = output_file_name + '-org.mp4'
        output_file_name_gen = output_file_name + '.avi'
        input_video_path = self._modalities_dict['video']['file_dict'][k][0]
        output_path = os.path.join(output_dir, output_file_name_org)
        copyfile(input_video_path, output_path)
        # try:
        #     #in case filename too long
        #     self.write_video_file(k, os.path.join(output_dir, output_file_name_gen))
        # except:
        #     self.write_video_file(k, os.path.join(output_dir, output_file_name_gen[:100]))
        self.write_video_dir(k, os.path.join(output_dir, output_file_name_gen))

    def mkdir_dir(self, dir_name):
        output_dir = os.path.join(self._data_folder, dir_name)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        return output_dir

    def write_video_dir(self, index, output_dir):
        modal = 'face'
        image_file_list = self._modalities_dict[modal]['file_dict'][index]
        image_logits = self._modal_data['data'][index]['modalities']['face_basic']['result']
        fps = self.extractors._frame_frequency  # 视频帧率
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        img12 = cv2.imread(image_file_list[0])
        image_size = (img12.shape[0], img12.shape[1])
        output_dir_per_video = output_dir
        if os.path.exists(output_dir_per_video):
            shutil.rmtree(output_dir_per_video)
        os.makedirs(output_dir_per_video)
        for i, image_file in enumerate(image_file_list):
            img12 = cv2.imread(image_file)
            output_file = os.path.join(output_dir_per_video, image_file.split('/')[-1])
            self._get_stastic(img12, basic_labels, image_logits[i],
                              [self._demo_display[modal]['x'],
                               self._demo_display[modal]['y']],
                              [self._demo_display[modal]['h'],
                               self._demo_display[modal]['w']],
                              modality=modal, get_frome_dict=False)
            cv2.imwrite(output_file, img12)

    def write_video_file(self, index, output_file):
        modal = 'face'
        image_file_list = self._modalities_dict[modal]['file_dict'][index]
        image_logits = self._modal_data['data'][index]['modalities']['face_basic']['result']
        fps = self.extractors._frame_frequency  # 视频帧率
        fps = 15
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        img12 = cv2.imread(image_file_list[0])
        image_size = (img12.shape[1], img12.shape[0])
        videoWriter = cv2.VideoWriter(output_file, fourcc, fps, image_size)

        for i, image_file in enumerate(image_file_list):
            img12 = cv2.imread(image_file)
            self._get_stastic(img12, basic_labels, image_logits[i],
                              [self._demo_display[modal]['x'],
                               self._demo_display[modal]['y']],
                              [self._demo_display[modal]['h'],
                               self._demo_display[modal]['w']],
                              modality=modal, get_frome_dict=False)
            videoWriter.write(img12)
        if image_file_list:
            img12 = cv2.imread(image_file)
            self._write_all_sta(img12, index)
            videoWriter.write(img12)
        videoWriter.release()

    def _write_all_sta(self, img12, index):
        for modal in self._modality_list + ['weight']:
            modal = modal.replace('video', 'ensemble')
            if modal not in ['weight']:
                labels = basic_labels
                logits = self._modal_data['data'][index]['modalities'][
                    self.ensemble_util.replace_map_final[modal]]['result'][0]
            else:
                weight_res = self._modal_data['data'][index]['modalities'][
                    self.ensemble_util.replace_map_final[modal]]['result']
                labels, logits = list(weight_res.keys()), list(weight_res.values())
            if modal in ['text']:
                labels = basic_labels + ['confidence']
                logits = list(logits) + [self._modal_data['data'][index]['modalities']['text']['result'][0][2][2]]
            self._get_stastic(img12, labels, logits,
                              [self._demo_display_static[modal]['x'],
                               self._demo_display_static[modal]['y']],
                              [self._demo_display_static[modal]['h'],
                               self._demo_display_static[modal]['w']],
                              modality=modal, get_frome_dict=False)

    def _demo(self):
        showBox = self._config['demo']['show_box']
        self._save_last_temp_at = {}
        video_captor = cv2.VideoCapture(0)
        self._audio_start = False  # 这个代表是否开始录音了，如果开始录音，则不现实audio的概率分布，并且按s键没有作用
        self._last_audio = -1  # 记录上一条录制的视频
        self._last_images_results = []  # 记录上一条录制的视频
        self.session_done = False  # 代表子进程是否完成，如果子进程完成，则可以开始预测
        self._save_last_result_modality = ['audio', 'text', 'ensemble']
        frameFrequency = 100
        while True:
            ret, self.frame = video_captor.read()
            frame = self.frame
            if frame is None:
                return
            self._show_face(frame, showBox)
            predict_res = self._predict_modality_sample(frame, frame, "face")
            face_logits, face_labels, frame = predict_res['logits'], predict_res['labels'], predict_res['frame']
            if not self._audio_start:
                cv2.putText(frame, "click 's' to start recording", (self._demo_display['note']['x'],
                                                                    self._demo_display['note']['y']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                if self.session_done:
                    # 录音停止的第一次刷新帧
                    audio_file = os.path.join(self._demo_folder, "audio", "%d.wav" % self._last_audio)
                    predict_res = self._predict_modality_sample(self.frame, audio_file, "audio")
                    audio_logits, audio_labels, frame = predict_res['logits'], predict_res['labels'], predict_res[
                        'frame']
                    predict_res = self._predict_modality_sample(self.frame, self._last_text, "text")
                    text_logits, text_labels, frame = predict_res['logits'], predict_res['labels'], predict_res['frame']
                    self._ensemble(audio_labels, audio_logits, face_labels, text_labels, text_logits)
                    # 可视化ensemble结果
                    self._get_stastic(frame, basic_labels, self._interactive_data['data']
                    ['interactive_last']['modalities']['ensemble_basic']['result'][0],
                                      [self._demo_display['ensemble']['x'],
                                       self._demo_display['ensemble']['y']],
                                      [self._demo_display['ensemble']['h'],
                                       self._demo_display['ensemble']['w']],
                                      modality='ensemble')
                    self.session_done = False  # 保证不重复预测
                    # 预测上一次录音过程的图像信息

                    # 对于Audio，Image，Text的结构进行ensemble
                    # 绘制雷达图
                elif self._last_audio != -1:
                    # 非录音状态刷新帧
                    for m in self._save_last_result_modality:
                        self._get_stastic(frame, *self._save_last_temp_at[m])  # 每次都绘制出上一次语音、文本, ensemble g识别的结果
                    # 非录音状态展示上一次的视频录制分析结果
                    # 绘制雷达图
                if cv2.waitKey(frameFrequency) & 0xFF == ord('s'):
                    # 开始录制声音刷新帧
                    self._audio_start = True  # 开始录制声音
                    self._last_audio = begin = time.time()  # 记录上一次语音识别结果
                    threading._start_new_thread(self.__recording, ())  # 开启一个新的进程来识别语音，保存录音
                    # 初始化记录dict
                    self.init_interactive_data()
            else:
                # 录音过程刷新帧
                cv2.putText(frame, "recording...", (self._demo_display['note']['x'],
                                                    self._demo_display['note']['y']),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)
                # 开始记录这一次的视频信息
                self._last_images_results.append([0, '', face_logits])

            # 这里是第一种录音方法，用的是pyaudio，上面用的是麦克风
            # if not self._audio_start:
            #     if self._last_audio != -1:
            #         self._get_stastic(frame, *self._save_last_temp_at["audio"])
            #     # start recording
            #     if cv2.waitKey(50) & 0xFF == ord('s'):
            #         self._audio_start = True
            #         rec = RecorderBack()
            #         begin = time.time()
            #         self._last_audio = begin
            #         print("Start recording")
            #         rec.start()
            # else:
            #     # stop recording and save to time.wav file
            #     if cv2.waitKey(50) & 0xFF == ord('e'):
            #         self._audio_start = False
            #         print("Stop recording")
            #         rec.stop()
            #         fina = time.time()
            #         t = fina - begin
            #         print('录音时间为%ds' % t)
            #         audio_file = os.path.join(self._demo_folder, "audio", "%d.wav" % self._last_audio)
            #         rec.save(audio_file)
            #         self._predict_modality_sample(frame, audio_file, "audio")
            cv2.imshow('ERFramework', frame)
            if cv2.waitKey(frameFrequency) & 0xFF == ord('q'):
                break

        video_captor.release()
        cv2.destroyAllWindows()

    def _ensemble(self, audio_labels, audio_logits, face_labels, text_labels, text_logits):
        self._interactive_data['info']['modal_labels']['audio'] = audio_labels
        self._interactive_data['info']['modal_labels']['text'] = text_labels
        self._interactive_data['info']['modal_labels']['face'] = face_labels
        self._interactive_data['data']['interactive_last']['modalities']['audio']['result'] = [[0, '', audio_logits]]
        self._interactive_data['data']['interactive_last']['modalities']['text']['result'] = [[0, '', text_logits]]
        self._interactive_data['data']['interactive_last']['modalities']['face']['result'] = self._last_images_results
        # ensemble
        self.ensemble_util.ensemble(self._interactive_data)

    def init_interactive_data(self):
        self._interactive_data = {'info': {
            'modal_labels': {
                'audio': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
                'text': ['positive_prob', 'negative_prob', 'confidence'],
                'face': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']},
            'used_modality_list': ['audio', 'video', 'face', 'text']},
            'data': {'interactive_last':
                         {'modalities':
                              {'audio': {},
                               'video': {},
                               'face': {},
                               'text': {}}}}}

    def __recording(self):
        r = sr.Recognizer()
        # 启用麦克风
        mic = sr.Microphone()
        print('录音中...')
        with mic as source:
            # 降噪
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        audio_file = os.path.join(self._demo_folder, "audio", "%d.wav" % self._last_audio)
        with open(audio_file, "wb") as f:
            # 将麦克风录到的声音保存为wav文件
            f.write(audio.get_wav_data(convert_rate=16000))
        print('录音结束，识别中...')
        fina = time.time()
        t = fina - self._last_audio
        print('录音时间为%ds' % t)
        target = audio_baidu(audio_file)
        self._last_text = target
        self.session_done = True
        self._audio_start = False

    def _show_face(self, frame, showBox):
        if showBox:
            detected_face, face_coor = format_image(frame)
            if face_coor is not None:
                [x, y, w, h] = face_coor
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def _predict_modality_sample(self, frame, data, modality):
        if isinstance(frame, str):
            frame = cv2.imread(frame)
        time_begin = time.time()
        print(time_begin)
        label, logits, labels = self.predictor[modality][self.predictor_select[modality][0]].predict_sample(data,
                                                                                                            return_all=True)
        time_end = time.time()
        if TEST_TIME:
            print('predict time', time_end - time_begin)
        time_begin = time_end
        target_name = 'Unknown'
        if modality == 'face' and "default" not in self._config['modalities']['face']['model_info']['model_name']:
            target_logits = np.zeros(len(labels))
            # 除了default模式下，其他的模式返回的是多个人脸,下面这里应该只展示目标用户的人脸情绪
            # logits : [[emotion_prediction, fname, face_coordinates],[emotion_prediction, fname, face_coordinates],...]
            fname = 'Unknown'
            for logit in logits:
                emotion_prediction, fname, face_coordinates = logit
                fname = fname.replace(' ', '')
                color, emotion_text = self.extract_color(emotion_prediction, labels)
                if fname == "Unknown":
                    name = emotion_text
                else:
                    name = str(fname) + " is " + str(emotion_text)
                draw_bounding_box(face_utils.rect_to_bb(face_coordinates), frame, color)
                draw_text(face_utils.rect_to_bb(face_coordinates), frame, name,
                          color, 0, -45, 1, 3)
                if self._target_name is None:
                    target_logits = emotion_prediction
                elif fname.lower() == self._target_name.lower():
                    target_logits = emotion_prediction
            logits = target_logits
            target_name = fname.lower()

        if logits is not None:
            self._get_stastic(frame, labels, logits,
                              [self._demo_display[modality]['x'],
                               self._demo_display[modality]['y']],
                              [self._demo_display[modality]['h'],
                               self._demo_display[modality]['w']],
                              modality=modality, get_frome_dict=False)
            time_end = time.time()
        return {'logits': logits, 'labels': labels, 'frame': frame, 'fname': target_name}

    def extract_color(self, emotion_prediction, labels):
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = labels[emotion_label_arg]
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        return color, emotion_text

    def _get_stastic(self, frame, labels, probs, xy, hw, modality, type=0, get_frome_dict=False):
        if isinstance(labels, dict):
            labels = list(labels.values())
        if not get_frome_dict:
            if self._interactive and modality in self._save_last_result_modality:
                self._save_last_temp_at[modality] = [labels, probs, xy, hw, modality, type]
        else:
            labels, probs, xy, hw, modality, type = self._save_last_temp_at[modality]
        if modality in ['face']:
            if self._face_smooth and self._last_face_logits is not None:
                probs += softmax(self._face_smooth_r * self._last_face_logits)
            self._last_face_logits = probs
        title = modality.upper()
        text_color = (255, 255, 255)
        text_color = (0, 0, 0)
        rectangle_color = (255, 0, 0)
        text_scale = 0.5
        x_offset, y_offset = xy
        title_height = 15
        h, w = hw
        class_number = len(labels)
        hh = h // class_number
        cv2.putText(frame, title, (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale + 0.2,
                    text_color, 2)
        y_offset += title_height
        for index, label in enumerate(labels):
            try:
                prob = probs[index]
            except Exception as e:
                print(probs)
                print(labels)
                print(e)
            if type == 0:
                ww = int(prob * w)
                text = "{}: {:.2f}%".format(label, prob * 100)
                rec_xy0 = (x_offset, (index * hh) + y_offset)
                rec_xy1 = (x_offset + ww, (index * hh) + hh + y_offset)
                text_xy0 = (x_offset + 3, (index * hh) + hh - hh // 5 + y_offset)
                cv2.rectangle(frame, rec_xy0,
                              rec_xy1, rectangle_color, -1)
                cv2.putText(frame, text, text_xy0,
                            cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                            text_color, 2)
            elif type == 1:
                cv2.rectangle(frame, (130, index * 20 + 10),
                              (130 + int(probs[index] * 100), (index + 1) * 20 + 4),
                              rectangle_color, -1)
                cv2.putText(frame, label, (10, index * 20 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            text_color, 2)

    def _predict(self, modal='audio', face_visualize=False):
        suffix = suffix_map[modal]
        self._modal_data['info']['modal_labels'] = {modal: self.predictor[modal][self.predictor_select[modal][0]].labels
                                                    for modal in list(set(self._modality_list) - {'video'})}
        all_input_data_count = 0.0
        bad_emotion_count = 0.0
        bad_face_count = 0.0
        # make the output dir
        output_dir = os.path.join(self._data_folder, 'face_output')
        bad_path_dict = {
            'emotion': os.path.join(output_dir, 'bad_emotion'),
            'face': os.path.join(output_dir, 'bad_facerec')
        }
        for k, v in list(bad_path_dict.items()):
            if os.path.exists(v):
                shutil.rmtree(v)
            os.makedirs(v, exist_ok=True)

        for index, value in self._modal_data['data'].items():
            label = value['label']
            if modal in value['modalities'].keys():
                file_list = self._modalities_dict[modal]['file_dict'][index]
                if face_visualize and modal in ['face']:
                    for file in file_list:
                        all_input_data_count += 1
                        if self._verbose:
                            print(all_input_data_count)
                        # 做了筛选，只剩下单人的情绪信息了
                        predict_res = self._predict_modality_sample(file, file, "face")
                        logits, labels, frame = predict_res['logits'], predict_res['labels'], predict_res['frame']
                        fname = predict_res['fname']
                        begin_time = time.time()
                        # 判断是情绪错了，还是face错了
                        if labels[np.argmax(logits)] != label[0]:
                            bad_emotion_count += 1
                            output_path = os.path.join(bad_path_dict['emotion'], file.split('/')[-1])
                            cv2.imwrite(output_path, frame)
                        if fname != label[1]:
                            bad_face_count += 1
                            output_path = os.path.join(bad_path_dict['face'], file.split('/')[-1])
                            cv2.imwrite(output_path, frame)
                        end_time = time.time()
                        if TEST_TIME:
                            print('write time', end_time - begin_time)
                else:
                    res = \
                        self.predictor[modal][self.predictor_select[modal][0]].predict(file_list, return_all=True)
                    value['modalities'][modal]['result'] = res[0]
        if face_visualize and modal in ['face']:
            print('accuracy: ')
            print('data count: ', all_input_data_count)
            print('acc of emotion: ', 1.0 - bad_emotion_count / all_input_data_count)
            print('acc of face: ', 1.0 - bad_face_count / all_input_data_count)
        # value['modalities'][modal]['labels'] = res[1]

    def _predict_all_modality(self, face_visualize=False):
        for modal in self._modality_list:
            if modal in ['video']:
                continue
            print('predict modal {}'.format(modal))
            self._predict(modal, face_visualize=face_visualize)

    # 提取出前缀对应的label，可以有不同的提取方式
    def _get_label(self):
        if self._config['dataset'] in ['ravdess']:
            return self._get_label_from_file_name(self._config['dataset'])
        elif self._config['dataset'] in ['friends']:
            return self._get_label_from_csv(self._config['dataset'], self._config['label_extract']['file_path'])
        elif self._config['dataset'] in ['aiic']:
            return self._get_label_from_file_name(self._config['dataset'])
        elif self._config['dataset'] in ['lovehouse']:
            return self._get_label_from_file_name(self._config['dataset'])
        else:
            return {index: -1 for index in self._modal_data['info']['index_list']}

    def _get_label_from_file_name(self, dataset):
        index_list = self._modal_data['info']['index_list']
        label_dict = {}
        for index in index_list:
            if dataset in ['ravdess']:
                label_dict[index] = [get_basic_word(
                    label_mapping[dataset][int(index.split('-')[2])]), -1]
            elif dataset in ['aiic']:
                # 情绪名称，人名
                label_dict[index] = [get_basic_word(index.split('_')[1]), index.split('_')[0]]
            elif dataset in ['lovehouse']:
                label_dict[index] = [get_basic_word(index.split('-')[2]), -1]
        return label_dict

    def _get_label_from_csv(self, dataset, filename):
        f = open(filename, encoding='UTF-8')
        data = read_csv(f)
        all_label_dict = {}
        label_dict = {}
        for i in range(len(data)):
            data_i = data.iloc[i]
            Dialogue_ID = data_i["Dialogue_ID"]
            Utterance_ID = data_i["Utterance_ID"]
            name_pattern = 'dia{}_utt{}'.format(Dialogue_ID, Utterance_ID)
            Emotion = data_i["Emotion"]
            all_label_dict[name_pattern] = Emotion
        index_list = self._modal_data['info']['index_list']
        for index in index_list:
            label_dict[index] = get_basic_word(all_label_dict[index])
        return label_dict

    def _print_extract_result(self):
        print('\n{} all modality file list'.format(self._output_prefix))
        print(self._output_prefix, self._modalities_dict)
        print('\n{} all modal data, including label, '
              'modality in every sample'.format(self._output_prefix))
        print(self._output_prefix, self._modal_data)
        print('\n{} all modality'.format(self._output_prefix))
        print(self._output_prefix, self._modality_list)
        print('\n{} label info'.format(self._output_prefix))
        print(self._output_prefix, self._label)
        # self.ensemble_util.ensemble(self._modal_data)

    def _construct_sample_data(self):
        self._get_modal_index()
        self._label = self._get_label()
        labels = self._label.values()
        labels = [l[0] for l in labels]
        self.labels_count = Counter(labels)
        self._get_sample_data()

    def _get_sample_data(self):
        index_list = self._modal_data['info']['index_list']
        modal_data = self._modal_data['data'] = {}
        self._modality_list = list(self._modalities_dict.keys())
        self._modal_data['info']['used_modality_list'] = self._modality_list
        for index in index_list:
            if len(index) == 0:
                continue
            # 判断是否所有模态都有相应数据，其中image需要区分前缀
            check_exist = [0] * len(self._modality_list)
            sample_modality = []
            for i, modality in enumerate(self._modality_list):
                if index in self._modalities_dict[modality]['file_dict'].keys():
                    check_exist[i] = 1
                    sample_modality.append(modality)
            # 要求的所有模态都存在
            if self._config['check_every_modal']:
                if np.all(check_exist):
                    modal_data[index] = {'label': self._label.get(index, -1),
                                         'modalities': {m: {} for m in self._modality_list}}
            else:
                modal_data[index] = {'label': self._label.get(index, -1),
                                     'modalities': {m: {} for m in sample_modality}}

    def _get_modal_index(self):
        # 根据不同模态数据路径，构建单个sample的不同模态数据路径，及标签数据
        self._modal_data = {'info': {}}
        # 使用视频或者声音数据进行索引，如果完全没有才使用图像数据，这个时候图像作为单独的数据，不与视频相关，所以没有影响
        index_modals = ['video', 'audio']
        index_list = []
        for index_modal in index_modals:
            if index_modal in self._modalities_dict:
                index_list = self._modalities_dict[index_modal]['file_dict']
        if len(index_list) == 0:
            # 使用图片名称作为索引
            index_modal = list(self._modalities_dict.keys())[0]
            index_list = self._modalities_dict[index_modal]['file_dict']
        self._modal_data['info']['index_modal'] = index_modal
        self._modal_data['info']['index_list'] = list(index_list.keys())

    def _extract_modality(self):
        # 对不同模态数据进行提取
        self._modalities_config_dict = self._config["modalities"]
        self._modalities_config_dict = self._config["modalities"]
        print(self._output_prefix)
        pprint.pprint(self._modalities_config_dict)
        # 用于保存modal的数据，包括不同模态的文件路径和标签信息
        self._modalities_dict = {}
        # video的path
        self._video_path = None
        if self._modalities_config_dict['video']['state'] == True \
                and \
                os.path.exists(
                    os.path.join(self._data_folder, 'video')):
            self._video_path = os.path.join(self._data_folder, 'video')
        # self._modalities_dict['audio'] = {}
        # 获取模态数据路径，其中有两种数据构造方法
        if self._config['caches']['modalities_dict']['state']:
            load_filename = os.path.join(root_path, self._config['caches']['modalities_dict']['filename'])
            with open(load_filename, 'rb') as handle:
                self._modalities_dict = pickle.load(handle)
                print('load modalities_dict data from {}'.format(load_filename))
                return
        for modal, modal_value in self._modalities_config_dict.items():
            # file_list 中保存的路径没有后缀
            modal_dir = os.path.join(self._data_folder, modal)
            if modal_value['state'] is True:  # 使用path中的路径提取数据
                if not os.path.exists(modal_dir):
                    raise ValueError("{} modal data state is true, "
                                     "but {} is not exsit".format(modal, modal_dir))
                file_dict = {modal_file.split('.')[0]: [os.path.join(modal_dir, modal_file)]
                             for modal_file in os.listdir(modal_dir) if not modal_file.startswith('.')}  # 这里可以增加标签
                # if self._config['dataset'] in ['lovehouse'] and modal in ['video']:
                #     file_dict = {k: v for k, v in file_dict.items() if
                #                  words2word.get(k.split('-')[-1], k.split('-')[-1]) in basic_labels}
                self._modalities_dict[modal] = {'file_dict': file_dict}
                if True:
                    print(modal)
                    print(modal_value['state'])
                    print(len(file_dict))

            elif modal_value['state'] == 'extract' and modal != 'video':  # 使用video中的数据进行提取
                if modal in ['text']:  # 从audio中提取文本的信息
                    file_dict = self.extractors.extract(modal, self._modalities_dict['audio'])
                    self._modalities_dict[modal] = {'file_dict': file_dict}
                    continue
                if self._video_path is None:
                    raise ValueError("u want extract {} data from video, "
                                     "but video state is not True or video "
                                     "path is mot exsit".format(modal))
                if os.path.exists(modal_dir):
                    shutil.rmtree(modal_dir)
                os.makedirs(modal_dir, exist_ok=True)
                # 这里从video中提取, 根据（modal, self._video_path），返回image paths list
                # 如果输出是图像的话，file_dict代表前缀的名称为key，真正的文件名list为value
                file_dict = self.extractors.extract(modal)
                self._modalities_dict[modal] = {'file_dict': file_dict}

            else:  # 不使用该模态
                pass
        with open(self._config['caches']['modalities_dict']['filename'], 'wb') as handle:
            pickle.dump(self._modalities_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _get_data_folder(self, interactive):
        # 判断数据文件是否存在
        self._data_folder = os.path.join(root_path, self._config["data_folder"])
        self._demo_folder = os.path.join(root_path, self._config["demo"]["data_folder"])
        os.makedirs(self._demo_folder, exist_ok=True)
        for modal in self._all_modality_list:
            os.makedirs(os.path.join(self._demo_folder, modal), exist_ok=True)
        if not os.path.exists(self._data_folder):
            if interactive:
                os.makedirs(self._data_folder)
            else:
                raise ValueError("data folder doesn't exits")


if __name__ == "__main__":
    w = Wrapper("{}/configs/basic.json".format(root_path), interactive=False, verbose=False)
