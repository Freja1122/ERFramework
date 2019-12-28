import os
import sys
import simplejson as json
import cv2
import numpy as np
import time

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
if root_path not in sys.path:
    sys.path.append(root_path)

import requests
import json
import base64
import os
import logging
import speech_recognition as sr

TEST_TIME = False

tookits_prefix = '[TOOKITS OUTPUT]'


def get_token():
    logging.info('开始获取token...')
    # 获取token
    baidu_server = "https://openapi.baidu.com/oauth/2.0/token?"
    grant_type = "client_credentials"
    client_id = "up7sdaBHdk09sbMk1l6ijszx"
    client_secret = "XmoFEcE4i8ErqBbnuSlgWb2B81AKXard"

    # 拼url
    url = f"{baidu_server}grant_type={grant_type}&client_id={client_id}&client_secret={client_secret}"
    res = requests.post(url)
    try:
        token = json.loads(res.text)["access_token"]
    except Exception as e:
        print(res.text)
    return token


def softmax(x):
    # Compute the softmax of vector x.
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

token = get_token()
def audio_baidu(filename):
    logging.info('开始识别语音文件...')
    with open(filename, "rb") as f:
        speech = base64.b64encode(f.read()).decode('utf-8')
    size = os.path.getsize(filename)
    headers = {'Content-Type': 'application/json'}
    url = "https://vop.baidu.com/server_api"
    data = {
        "format": filename[-3:],
        "rate": "16000",
        "dev_pid": "1537",
        "speech": speech,
        "cuid": "TEDxPY",
        "len": size,
        "channel": 1,
        "token": token,
    }

    req = requests.post(url, json.dumps(data), headers)
    result = json.loads(req.text)
    time.sleep(5)#sleep
    if result["err_msg"] == "success.":
        print('{} {} {}'.format(tookits_prefix, filename, result['result']))
        return result['result']
    else:
        print(tookits_prefix + filename + "内容获取失败，退出语音识别")
        return ['']


CASC_PATH = os.path.join(root_path,
                         'pretrained_models/face/'
                         'haarcascade_files/haarcascade_frontalface_default.xml')
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if not len(faces) > 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print(tookits_prefix + "[+} Problem during resize")
        return None, None
    return image, face_coor


convert_dict = {"none": None, "true": True, "false": False}


def convert_kwargs(kwargs):
    if not isinstance(kwargs, dict):
        return convert_dict.get(kwargs, kwargs)
    for k, v in kwargs.items():
        if isinstance(v, dict):
            convert_kwargs(v)
        else:
            kwargs[k] = convert_dict.get(str(v), v)
    return kwargs


def get_config(config):
    if isinstance(config, str):
        with open(config, "r") as f:
            config = convert_kwargs(json.load(f))
    elif isinstance(config, dict):
        config = config
    else:
        raise ValueError('config must be file path or dict')
    return config


suffix_map = {
    'audio': '.wav',
    'face': '.jpg',
    'pose': '.jpg',
    'text': '.txt'
}

label_mapping = {
    "ravdess": {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
    },
    'baidu_text': {
        0: 'positive_prob',
        1: 'negative_prob',
        2: 'confidence'
    },
    "fer2013": {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'}
}

word2words = {
    'neutral': ["neutral"],
    'calm': ["calm"],
    'happy': ["happy", 'joy'],
    "sad": ["sad", 'sadness'],
    "angry": ["angry", 'anger'],
    "fearful": ["fearful", 'fear'],
    "disgust": ["disgust"],
    "surprised": ["surprised", 'surprise']
}
basic_labels = ['calm', 'happy', 'sad', 'angry', 'surprised']

word2basicemotion = {
    'neutral': ['calm'],
    'calm': ['calm'],
    'happy': ["happy"],
    "sad": ["sad"],
    "angry": ["angry"],
    "fearful": ["sad"],
    "disgust": ["sad"],
    "surprised": ["surprised"],
    "positive_prob": ['happy'],
    "negative_prob": ['sad','angry'],
    "confidence": None
}

words2word = {vv: k for k, v in word2words.items() for vv in v}


def get_basic_word(word):
    return words2word.get(word, word)
