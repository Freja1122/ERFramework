import os
import sys
import cv2
import subprocess

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
if root_path not in sys.path:
    sys.path.append(root_path)

from utils import *

class ModalityExtractor:
    def __init__(self, config, interactive, verbose):
        self._config = config
        self._interactive = interactive
        self._verbose = verbose
        self._output_prefix = ['\t [Modality Extractor] ']
        self._data_folder = os.path.join(root_path, self._config["data_folder"])
        if self._config['modalities']['video']['state']:
            self._video_folder = os.path.join(self._data_folder, 'video')
            self._video_files = os.listdir(self._video_folder)
            self._video_suffixs = self._config["modalities"]['video']['suffix']
            self._frame_frequency = self._config['modalities']['video']['frames_cut']
        # 构建所有输出文件
        self._all_modality_list = [modal for modal in list(self._config['modalities'].keys())]
        # self._output_dir_dict = {}
        # 在wrapper里面已经创建过文件夹了
        # for modal in self._all_modality_list:
        #     self._output_dir_dict[modal] = os.path.join(self._data_folder, modal)
        #     os.makedirs(self._output_dir_dict[modal], exist_ok=True)

    def extract(self, modality, audio_for_text=None):
        output_dir = os.path.join(self._data_folder, modality)
        if modality in ['audio']:
            return self.extract_audio(output_dir)
        elif modality in ['face']:
            return self.extract_face(output_dir)
        elif modality in ['text']:
            return self.extract_text(output_dir, audio_for_text)

    def extract_audio(self, output_dir):
        video_files = self._video_files
        output_dict = {}
        for video_file in video_files:
            file_without_fix = self._get_video_file_without_fix(video_file)
            if file_without_fix is None:
                continue
            file_path = os.path.join(self._video_folder, video_file)
            output_path = os.path.join(output_dir, file_without_fix + '.wav')
            output_dict[file_without_fix] = [output_path]
            # command = "/Users/yuannnn/anaconda3/bin/ffmpeg -i {} -ab 160k -ac 2 -ar 16000 -vn {}".format(file_path, output_path)
            # PCM:ffmpeg -i test.mp4 -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -vn test.pcm
            # command = f"ffmpeg -i {file_path} -acodec pcm_s16le -f s16le -ac 1 -ar 16000 -vn {output_path}"
            # WAV:ffmpeg -i test.mp4 -ac 1 -ar 16000 -vn test.wav
            command = "ffmpeg -i {} -ac 1 -ar 16000 -vn {}".format(file_path, output_path)
            if self._verbose:
                print(self._output_prefix, 'command')
                print(self._output_prefix, command)
                print(self._output_prefix, file_path, '--- to --->', output_path)
            # out_bytes = subprocess.check_call(command, shell=True)
            out_bytes = subprocess.call(command, shell=True)
            if self._verbose:
                print(self._output_prefix, 'out_bytes', out_bytes)
        return output_dict

    def extract_face(self, output_dir):
        video_files = self._video_files
        frameFrequency = self._frame_frequency
        file_frame_dict = {}  # {没有prefix的文件名：[生成的同一个视频不同帧的文件名]}
        for video_file in video_files:
            file_without_fix = self._get_video_file_without_fix(video_file)
            if file_without_fix is None:
                continue
            file_frame_dict[file_without_fix] = []
            times = 0
            video_path = os.path.join(self._video_folder, video_file)
            camera = cv2.VideoCapture(video_path)
            length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._verbose:
                print(self._output_prefix, 'video length', length)
            while True:
                times += 1
                res, image = camera.read()
                if not res:
                    print(self._output_prefix, 'not res, not image')
                    break
                if times % frameFrequency == 0:
                    output_filename = os.path.join(output_dir, '{}_{}.jpg'.format(file_without_fix, times))
                    cv2.imwrite(output_filename, image)
                    file_frame_dict[file_without_fix].append(output_filename)
        return file_frame_dict

    def _get_video_file_without_fix(self, video_file):
        if video_file.startswith('.'):
            return None
        file_without_fix = video_file
        for _video_suffix in self._video_suffixs:
            file_without_fix = file_without_fix.replace(_video_suffix, '')
        return file_without_fix

    def extract_text(self, output_dir, audio_for_text):
        texts = {}
        for k, audio in list(audio_for_text['file_dict'].items()):
            text = audio_baidu(audio[0])
            texts[k] = text
        return texts


print()
