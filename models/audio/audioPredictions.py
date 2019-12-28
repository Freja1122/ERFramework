import keras
import numpy as np
import librosa
import pickle
import numpy as np

import os
import sys

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


class AudioPredictions:

    def __init__(self, _config, verbose=False):
        self._config = _config['modalities']['audio']['model_info']['default']
        path = os.path.join(root_path, self._config['pretrained_model'])
        self.path = path
        # self.file = file
        self._verbose = verbose
        self.load_model()
        self._modality = 'audio'
        self.labels = list(label_mapping['ravdess'].values())

    def load_model(self):
        '''
        I am here to load you model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        '''
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self):
        '''
        I am here to process the files and create your features.
        '''
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=2)
        x = np.expand_dims(x, axis=0)
        # predictions = self.loaded_model.predict_classes(x)
        result = self.loaded_model.predict(x)
        predictions = result.argmax()
        if self._verbose:
            print("Prediction is", " ", label_mapping['ravdess'][predictions])
        return predictions, result


    def predict_sample(self, file_path, return_all=False):
        self.file = file_path
        pred_res, predictions = self.makepredictions()
        if return_all:
            return pred_res, predictions[0], self.labels
        else:
            return pred_res, predictions[0], None

    def predict(self, path_label_audio, return_all=False):
        predict_label, _predict_label = [], []
        predict_logits, _predict_logits = [], []
        for i, pl in enumerate(path_label_audio):
            inpath, label = pl, -1
            if self._verbose:
                print()
                print(i)
            result, predictions, labels = self.predict_sample(inpath, return_all=return_all)
            if result is not None:
                _predict_label.append([inpath, label, result, label_mapping['ravdess'][result]])  # 路径、标签、结果
                _predict_logits.append([inpath, label, predictions])  # 路径、标签、logits
                predict_label.append([result, label_mapping['ravdess'][result], predictions])
                if self._verbose:
                    print(_predict_label[-1])
                    print(_predict_logits[-1])
        if return_all:
            return predict_label,labels
        return predict_label


# Here you can replace path and file with the path of your model and of the file from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.
if __name__ == '__main__':
    pred = AudioPredictions(path=os.path.join(root_path,
                                              'pretrained_models/audio/Emotion_Voice_Detection_Model.h5'), file='')

    # pred.load_model()
    pred._predict([['/data1/bixiao/Code/ERFramework/data/friends/audio/dia2_utt5.wav', 0]])
