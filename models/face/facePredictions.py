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
        self.input = config['input']
        self.conf_threshold = config['conf_threshold']
        self.fps = config['fps']
        self.show = config['show']
        self.model = os.path.join(root_path, config['pretrained_model'])
        self.modelFile = os.path.join(root_path, config['model_file'])
        self.configFile = os.path.join(root_path, config['config_file'])
        self.face_lamdmark_model = os.path.join(root_path, config['face_lamdmark_model'])
        self.output_dir = os.path.join(root_path, _config['data_folder'], 'face_out')
        self.labels = list(label_mapping['ravdess'].values())
        self._load_model()

    def predict_sample(self, data, return_all=False):
        if isinstance(data, str):
            image = cv2.imread(data, 1)
            if image is None:
                print("Can't find {} image".format(data))
                # sys.exit(-1)
        else:
            image = data
        success, image, emotion_predictions = self.detect_emotions(image, self.emotion_detector, self.face_detector)
        label = np.argmax(emotion_predictions)
        if return_all:
            return label, emotion_predictions, self.labels
        else:
            return label, emotion_predictions, None

    def predict(self, data, return_all=False):
        # read image
        res = []
        for i, data_list in enumerate(data):
            source, label = data_list, -1
            label, emotion_predictions, labels = self.predict_sample(source, return_all)
            res.append([label, label_mapping['ravdess'][label], emotion_predictions])
        if return_all:
            return res, labels
        return res

    def detect_emotions(self, image, emotion_detector, face_detector):
        ''' Detects emotion on image

        Args:
            image (array): image to detect emotions on
            self.emotion_detector: loaded model for face expression classification
            self.face_detector: loaded model for face detection
            self: command arguments

        Returns:
            bool: True if process was successfull
            image (array): input image after processing
        '''
        global images

        # detect faces
        success, faces = self.get_faces(image, self.face_detector)
        emotion_predictions = np.zeros(8)
        emotion_predictions[0] = 1
        if (success):

            # loop through all found faces
            # for f in range(faces.shape[2]):
            for f in range(1):
                confidence = faces[0, 0, f, 2]
                if confidence > self.conf_threshold:
                    x1 = int(faces[0, 0, f, 3] * image.shape[1])
                    y1 = int(faces[0, 0, f, 4] * image.shape[0])
                    x2 = int(faces[0, 0, f, 5] * image.shape[1])
                    y2 = int(faces[0, 0, f, 6] * image.shape[0])

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # detected_face = fa.align(image, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                    detected_face = image[y1:y2, x1:x2]
                    if detected_face.size != 0:

                        # resize, normalize and save the frame (convert to grayscale if frames_resolution[-1] == 1)
                        if (self.emotion_detector.input_shape[-1] == 1):
                            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

                        detected_face = cv2.resize(detected_face,
                                                   (self.emotion_detector.input_shape[-3],
                                                    self.emotion_detector.input_shape[-2]))
                        detected_face = cv2.normalize(detected_face, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                      dtype=cv2.CV_32F)

                        # ___ one image emotion detector
                        if len(self.emotion_detector.input_shape) == 4:
                            detected_face = np.expand_dims(detected_face, axis=0)

                            # if emotion detector input is grayscale image
                            if self.emotion_detector.input_shape[-1] == 1:
                                detected_face = np.expand_dims(detected_face, axis=3)
                            emotion_predictions = self.detect_emotion_image(detected_face, self.emotion_detector)
                            image = self.draw_results(image, emotion_predictions, [x1, y1, x2, y2])

                        elif self.input == "image":
                            print("This emotion detector does not support emotion classification on 1 image")
                            success = False
                            return success, image

                        # ___ multiple image emotion detector
                        if len(self.emotion_detector.input_shape) == 5:
                            if self.emotion_detector.input_shape[-1] == 1:
                                detected_face = np.expand_dims(detected_face, axis=4)
                            images.append(detected_face)

                            # if enough images in storage
                            if self.emotion_detector.input_shape[1] == len(images):
                                images_arr = np.expand_dims(np.asarray(images), axis=0)
                                emotion_predictions = self.detect_emotion_video(images_arr, self.emotion_detector)
                                image = self.draw_results(image, emotion_predictions, [x1, y1, x2, y2], self)
                                images = list(images[-29:])
        else:
            print("Unsuccessfull image processing")
            success = False

        return success, image, emotion_predictions

    def get_faces(self, image, face_detector):
        ''' Get faces information from image

        Args:
            image (array): Image to process
            face_detector: loaded model for face detection

        Returns:
            bool: True if at least 1 face was found
            array of vectors: one vector for each face (array of [0,0,confidence,x1,y1,x2,y2])
        '''

        success = True
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        face_detector.setInput(blob)
        faces = face_detector.forward()

        if (faces.shape[2] == 0):
            print("No faces were found")
            success = False
        return success, faces

    def detect_emotion_image(self, face, emotion_detector):
        ''' Get emotion prediction from image

        Args:
            face (array): Face picture to process
            emotion_detector: loaded model for face expression classification

        Returns:
            vector: confidence for each emotion
        '''

        emotion_predictions = emotion_detector.predict(face)[0]
        return emotion_predictions

    def draw_results(self, image, emotion_predictions, coords):
        ''' Put on the image rectangle around face and text about it's emotion

        Args:
            image (array): to put rectangle and text on
            emotion_predictions: vector of confidences for each emotion
            coords: [x1, y1, x2, y2] vector of face location

        Returns:
            image (array): input image after processing
        '''

        emotion_probability = np.max(emotion_predictions)
        emotion_label_arg = np.argmax(emotion_predictions)
        # show result
        if (self.show):
            if emotion_label_arg == 0:  # neutral
                color = emotion_probability * np.asarray((100, 100, 100))
            elif emotion_label_arg == 1:  # calm
                color = emotion_probability * np.asarray((100, 100, 100))
            elif emotion_label_arg == 2:  # happy
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_label_arg == 3:  # sad
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_label_arg == 4:  # angry
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_label_arg == 5:  # fearful
                color = emotion_probability * np.asarray((100, 100, 100))
            elif emotion_label_arg == 6:  # disgust
                color = emotion_probability * np.asarray((100, 100, 100))
            elif emotion_label_arg == 7:  # surprized
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), color, 1)

            # write emotion text above rectangle
            emotion_percent = str(np.round(emotion_probability * 100))
            cv2.putText(image, emotion_percent + "%", (coords[0], coords[3] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            emotion_en = label_mapping['ravdess'][emotion_label_arg]
            # emotion_ru = emotions_ru[emotion_label_arg]
            cv2.putText(image, emotion_en, (coords[0], coords[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return image

    def _load_model(self):
        # ___ FACE DETECTOR MODEL ___
        self.face_detector = cv2.dnn.readNetFromTensorflow(self.modelFile, self.configFile)
        # ___ EMOTION RECOGNITION MODEL ___
        self.emotion_detector = None
        if os.path.isfile("{}.json".format(self.model)):
            json_file = open("{}.json".format(self.model), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.emotion_detector = model_from_json(loaded_model_json)
            self.emotion_detector.load_weights("{}.h5".format(self.model))

        elif os.path.isfile("{}.hdf5".format(self.model)):
            self.emotion_detector = load_model("{}.hdf5".format(self.model), compile=False)
        else:
            print("Error. File {} was not found!".format(self.model))
            # sys.exit(-1)
        frames_resolution = [self.emotion_detector.input_shape[-3], self.emotion_detector.input_shape[-2],
                             self.emotion_detector.input_shape[-1]]
        # ___ FACE ALIGNER ___ (uses emotion recognition model input shape)
        predictor = dlib.shape_predictor(self.face_lamdmark_model)
        fa = FaceAligner(predictor, desiredFaceWidth=self.emotion_detector.input_shape[-3])
        # output file path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # name = (os.path.normpath(self.source.replace('../', '')).split("/")[-1] if (self.input != "camera") else ".avi")
        # save_path = os.path.join(self.output_dir, name)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.VideoWriter_fourcc(*'MPEG')
        # vid = None
        end_load = time.time()
        return end_load


if __name__ == '__main__':
    config = get_config("{}/configs/basic.json".format(root_path))
    face_predictor = FacePredictions(config)
    res = face_predictor.predict([
        ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_14.jpg', 0],
        ['/data1/bixiao/Code/ERFramework/data/friends/face/dia2_utt5_21.jpg', 0]
    ])
    print(res)
