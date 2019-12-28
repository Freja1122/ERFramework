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
import face_recognition
from imutils import face_utils
import requests
from utils import *

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
from utils.face.datasets import get_labels
from utils.face.inference import detect_faces
from utils.face.inference import draw_text
from utils.face.inference import draw_bounding_box
from utils.face.inference import apply_offsets
from utils.face.inference import load_detection_model
from utils.face.preprocessor import preprocess_input


class FacePredictions_faceandemotion:
    def __init__(self, _config, verbose=False):
        config = _config['modalities']['face']['model_info']['face_and_emotion']
        self._config = config
        self.labels = list(get_labels('fer2013').values())
        self._emotion_offsets = (20, 40)
        self._known_face_encodings = []
        self._known_face_names = []
        self._open_face_images = []
        self._load_model()

    def detect_emotions(self, image, process_this_frame=True, only_logits=False):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        begin_time = time.time()
        faces = self._detector(rgb_image)
        end_time = time.time()
        if TEST_TIME:
            print('face detector time: ', end_time - begin_time)
        begin_time = time.time()
        face_name = self.face_compare(rgb_image, process_this_frame)
        end_time = time.time()
        if TEST_TIME:
            print('face compare time: ', end_time - begin_time)
        emotion_predictions = []
        for face_coordinates, fname in zip(faces, face_name):
            x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), self._emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (self._emotion_target_size))
            except:
                continue
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            begin_time = time.time()
            emotion_prediction = self._emotion_classifier.predict(gray_face)[0]
            end_time = time.time()
            if TEST_TIME:
                print('predict sample time: ', end_time - begin_time)
            emotion_predictions.append([emotion_prediction, fname, face_coordinates])
        if not emotion_predictions:
            emotion_predictions = [[np.zeros(len(self.labels)), "Unknown", [(0, 0),(0, 0)]]]
        return emotion_predictions

    def predict_sample(self, data, return_all=False, process_this_frame=True):
        if isinstance(data, str):
            image = cv2.imread(data, 1)
            if image is None:
                print("Can't find {} image".format(data))
                # sys.exit(-1)
        else:
            image = data
        emotion_predictions = self.detect_emotions(image, process_this_frame)
        if return_all:
            return -1, emotion_predictions, self.labels
        else:
            return -1, emotion_predictions, None

    def predict(self, data, return_all=False):
        # read image
        res = []
        for i, data_list in enumerate(data):
            source = data_list
            label, emotion_predictions, labels = self.predict_sample(source, return_all)
            # 这里因为检测的是多张人脸，暂时没办法复用之前的接口，所以埋个bug
            res.append([label, None, emotion_predictions])
        if return_all:
            return res, labels
        return res

    def _load_model(self):
        emotion_model_path = self._config['emotion_model_path']
        # loading models
        self._detector = dlib.get_frontal_face_detector()
        self._emotion_classifier = load_model(emotion_model_path)

        self._predictor = dlib.shape_predictor(self._config['face_lamdmark_model'])

        # getting input model shapes for inference
        self._emotion_target_size = self._emotion_classifier.input_shape[1:3]

        # for dictionary in data/face/knownface
        if self._config.get('know_face_dir', None):
            know_faces_files = os.listdir(self._config['know_face_dir'])
            for know_faces_file in know_faces_files:
                self._load_face_rec_image(os.path.join(root_path, self._config['know_face_dir'], know_faces_file))
        print(self._known_face_names)

    def _load_face_rec_image(self, imagepath):
        # Load a sample picture and learn how to recognize it.

        person_image = face_recognition.load_image_file(imagepath)
        person_face_encoding = face_recognition.face_encodings(person_image)[0]
        self._known_face_encodings.append(person_face_encoding)
        person_name = imagepath.split('/')[-1].replace('.jpg', '')
        self._known_face_names.append(person_name)
        self._open_face_images.append(open(imagepath, 'rb'))

    def face_compare(self, frame, process_this_frame, compare_tec='dlib'):
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            if compare_tec == 'dlib':
                # Find all the faces and face encodings in the current frame of video
                begin_time = time.time()
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                end_time = time.time()
                if TEST_TIME:
                    print('face_encodings: ', end_time - begin_time)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    begin_time = time.time()
                    matches = face_recognition.compare_faces(self._known_face_encodings, face_encoding,
                                                                          tolerance=0.4)
                    end_time = time.time()
                    if TEST_TIME:
                        print('compute distance: ', end_time - begin_time)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = self._known_face_names[first_match_index]

                    face_names.append(name)
            elif compare_tec == 'face++':
                self.check_face__()

        process_this_frame = not process_this_frame

        return face_names

    def check_face__(self, curr_face):
        time = time.time()
        filepath = os.path.join(root_path, 'data', 'temppng', time + '.png')
        with open(filepath, 'wb') as f:
            f.write(curr_face)
        compare_url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
        key = "XSyHeF1ysKH4dpgiuRUvNydxB4pzJMp8"
        secret = "4-nmA0PI-xij4nRVQ2RVOrFlNzoDVpGa"
        data = {'api_key': key, 'api_secret': secret}
        for open_face in self._open_face_images:
            files = {
                "iamge_file1": open_face,
                "image_file2": open(filepath, 'rb')
            }
            response = requests.post(compare_url, data=data, files=files)
            req_con = response.content.decode('utf-8')
            req_dict = JSONDecoder().decode(req_con)
        return


if __name__ == '__main__':
    config = get_config("{}/configs/basic.json".format(root_path))
    face_predictor = FacePredictions(config)
    res = face_predictor.predict([
        ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_14.jpg', 0],
        ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_21.jpg', 0]
    ])
    print(res)
