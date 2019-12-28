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
from imutils.face_utils import FaceAligner
from tensorflow.keras.models import load_model, model_from_json

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


class FacePredictions:
    def __init__(self, _config, verbose=False):
        config = _config['modalities']['face']['model_info']['default']
        self._config = config

    def predict_sample(self, data, return_all=False):
        if isinstance(data, str):
            image = cv2.imread(data, 1)
            if image is None:
                print("Can't find {} image".format(data))
                # sys.exit(-1)
        else:
            image = data
        success, image, emotion_predictions = self.detect_emotions(image)
        label = np.argmax(emotion_predictions)
        if return_all:
            return label, emotion_predictions, self._labels
        else:
            return label, emotion_predictions

    # def detect_emotions(self):

    def predict(self, data):
        # read image
        res = []
        for i, data_list in enumerate(data):
            source, label = data_list, -1
            label, emotion_predictions = self.predict_sample(source)
            res.append([label, label_mapping['ravdess'][label], emotion_predictions])
        return res


if __name__ == '__main__':
    # config = get_config("{}/configs/basic.json".format(root_path))
    # face_predictor = FacePredictions(config)
    # res = face_predictor.predict([
    #     ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_14.jpg', 0],
    #     ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_21.jpg', 0]
    # ])
    # print(res)
    # -*- coding: utf-8 -*-
    import urllib.request
    import urllib.error
    import time

    http_url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    key = "XSyHeF1ysKH4dpgiuRUvNydxB4pzJMp8"
    secret = "4-nmA0PI-xij4nRVQ2RVOrFlNzoDVpGa"
    filepath = os.path.join(root_path, "data/ravdess/face/01-01-07-01-01-01-02.png")

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
    data.append(key)
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
    data.append(secret)
    data.append('--%s' % boundary)
    fr = open(filepath, 'rb')
    data.append('Content-Disposition: form-data; name="%s"; filename=" "' % 'image_file')
    data.append('Content-Type: %s\r\n' % 'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_landmark')
    data.append('1')
    data.append('--%s' % boundary)
    data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
    data.append(
        "gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus")
    data.append('--%s--\r\n' % boundary)

    for i, d in enumerate(data):
        if isinstance(d, str):
            data[i] = d.encode('utf-8')

    http_body = b'\r\n'.join(data)

    # build http request

    req = urllib.request.Request(url=http_url, data=http_body)

    # header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)

    try:
        # post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # get response
        qrcont = resp.read()
        # if you want to load as json, you should decode first,
        # for example: json.loads(qrount.decode('utf-8'))
        import pprint
        pprint.pprint(qrcont.decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(e.read().decode('utf-8'))
