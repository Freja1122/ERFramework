import os
import sys
import simplejson as json
import cv2
import numpy as np
import shutil
from shutil import copyfile

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
if root_path not in sys.path:
    sys.path.append(root_path)


def get_file_list(dir_name):
    file_list = os.listdir(dir_name)
    return [f for f in file_list if not f.startswith('.')]

# 把数据从文件夹中读出来，然后变成 name_emotion_id.jpg
data_input_dir = os.path.join(root_path, 'data/aiic/aiic_dataset')
output_dir = os.path.join(root_path, 'data/aiic_input/face')
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)
name_list = get_file_list(data_input_dir)
index = 0
for name in name_list:
    input_name_dir = os.path.join(data_input_dir, name)
    emotion_file_list = get_file_list(input_name_dir)
    for emotion_file in emotion_file_list:
        index+=1
        input_emotion_file_path = os.path.join(input_name_dir, emotion_file)
        new_file_name = '{}_{}'.format(name, emotion_file)
        output_path = os.path.join(output_dir, new_file_name)
        copyfile(input_emotion_file_path, output_path)

print(index)