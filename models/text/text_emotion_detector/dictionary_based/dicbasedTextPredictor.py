#! /usr/bin/env python
# encoding: utf-8
import os
import sys
from importlib import reload
from models.text.text_emotion_detector.parser.preprocess import *
from models.text.text_emotion_detector.utils.dictionary_preprocess import *
from sklearn import metrics
from sklearn.metrics import classification_report

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


class DicbasedTextPredictor:
    def __init__(self, config={}, verbose=True):
        self.config = config
        self.verbose = verbose
        self.emotions = []
        self.text_preprocess = TextPreprocess()
        self.emotion_dictionary = EmotionDictionary()
        # part 1:情感词典录入
        self.words_dictionary = {
            "positive_emotion": TextPreprocess.read_text_file("positive-emotion.txt"),
            "negative_emotion": TextPreprocess.read_text_file("positive-emotion.txt"),
            "emotion_dictionary": self.emotion_dictionary.dictionary,
            "emotion_labels": EmotionDictionary.emotion_map.keys(),
            "extreme": TextPreprocess.read_text_file("extreme-6.txt"),
            "very": TextPreprocess.read_text_file("very-5.txt"),
            "more": TextPreprocess.read_text_file("more-4.txt"),
            "alittlebit": TextPreprocess.read_text_file("alittlebit-3.txt"),
            "insufficiently": TextPreprocess.read_text_file("insufficiently-2.txt"),
            "over": TextPreprocess.read_text_file("over-1.txt"),
            "no": TextPreprocess.read_text_file("no.txt"),
            "reverse_emotion": self.emotion_dictionary.reverse_emotion
        }

    def get_emotion_score(self, sentence_org, use_extend):
        sentence = self.text_preprocess.preprocess(sentence_org)
        if self.verbose:
            print(sentence)
        emotion_scores = {e: {'value': 0, 'extend': 1,
                              'not': 0, 'final': 0,
                              'value_extend': 0} for e in
                          self.words_dictionary['emotion_labels']}
        emotion_dictionary = self.words_dictionary['emotion_dictionary']
        same_weight = 1
        modi_weight = {
            "extreme": 2,
            "very": 1.4,
            "more": 1,
            "alittlebit": 0.4,
            "insufficiently": -0.2,
            "over": 1.2
        }
        for words in sentence:
            word, cixing = words
            # 情绪词典分数计算
            if word in emotion_dictionary.keys():
                word_emotion = emotion_dictionary[word]['emotion'][0]
                emotion_scores[word_emotion]['value'] += same_weight
                emotion_scores[word_emotion]['value_extend'] += emotion_dictionary[word]['extend']
            # 修饰词分数计算
            for modi in modi_weight.keys():
                if word in self.words_dictionary[modi]:
                    for e in emotion_scores.keys():
                        emotion_scores[e]['extend'] += modi_weight[modi]
                    break
            # 副词处理
            if word in self.words_dictionary['no']:
                for e in emotion_scores.keys():
                    emotion_scores[e]['not'] += 1
        for k, v in emotion_scores.items():
            if use_extend:
                emotion_value = v['value_extend']
            else:
                emotion_value = v['value']
            if v['not'] != 0:
                # 如果有not，对于最高的情绪做反转
                for reverse_emotion in self.words_dictionary['reverse_emotion'][k]:
                    emotion_scores[reverse_emotion]['final'] += emotion_value * v['extend']
            else:
                emotion_scores[k]['final'] = emotion_value * v['extend']

        finals = {e: emotion_scores[e]['final'] for e, v in emotion_scores.items()}
        if self.verbose:
            print(sentence_org)
            print(finals)
        return finals

    def _get_emotion_from_text(self, sentence_org, use_extend=False):
        emotion_score = self.get_emotion_score(sentence_org, use_extend=use_extend)
        emotion_labels = list(emotion_score.keys())
        emotion_idx = np.argmax(list(emotion_score.values()))
        emotion_score_sample = list(emotion_score.values())[emotion_idx]
        emotion = emotion_labels[emotion_idx]
        return emotion, emotion_score_sample


def construct_test_data(test_data_path):
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
        last_emotion_label = None
        data_emotion_number = {e: 0 for e in EmotionDictionary.emotion_map.keys()}
        all_data = {}
        for l in lines:
            l = l.strip()
            if '$' in l:
                emotion_label = l.split('$')[1]
            elif l:
                all_data[l] = {'label': emotion_label}
                data_emotion_number[emotion_label] += 1
            else:
                continue
        return all_data, data_emotion_number


if __name__ == "__main__":
    dic = DicbasedTextPredictor()
    all_data, data_emotion_number = construct_test_data('../data/test/test_set.csv')
    # dic.get_emotion_score('太难了')
    # dic.get_emotion_score('我好难啊')
    pred_array, label_array, acc_array = [], [], []
    labels = list(EmotionDictionary.emotion_map.keys())
    emotion_to_index = {e: i for i, e in enumerate(labels)}
    emotion_changed_map = {'good': 'happy'}
    print(data_emotion_number)
    all_data_keys = all_data.keys()
    test_score = True
    test_sample = False
    if test_sample:
        all_data_keys = ['上台帮男生打领带还说没什么，但是居然被主持人叫上去玩第二论游戏就超级郁闷']
        all_data = {k:{} for k in all_data_keys}
        test_score = False
    with open('bad_res1.txt', 'w') as f:
        for d in all_data_keys:
            emotion, emotion_score = dic._get_emotion_from_text(d, True)
            if emotion in ['good']:
                emotion = emotion_changed_map.get(emotion, emotion)
            all_data[d]['predict'] = emotion
            all_data[d]['predict_prob'] = emotion_score
            pred_array.append(emotion_to_index[emotion])
            label_array.append(emotion_to_index[all_data[d].get('label','neutral')])
            right = emotion == all_data[d].get('label','neutral')
            acc_array.append(right)
            if test_sample:
                print('predict label', emotion)
            if not right:
                f.write(' : '.join([all_data[d].get('label','neutral'), emotion, d]))
                f.write('\n')

    if test_score:
        print('=' * 100)
        print('7 classes')
        precision = metrics.precision_score(label_array, pred_array, average='micro')
        recall = metrics.recall_score(label_array, pred_array, average='micro')
        f1_score = metrics.f1_score(label_array, pred_array, average='weighted')
        label_array, pred_array = np.array(label_array), np.array(pred_array)
        print('precision', precision)
        print('precision_me', float(np.sum(acc_array)) / float(len(label_array)))
        print('recall', recall)
        print('f1_score', f1_score)
        labels_write = labels.copy()
        labels_write.remove('good')
        print(classification_report(label_array, pred_array, target_names=labels_write))

        print('=' * 100)
        print('5 classes')
        labels_index = []
        map_words = {
            'disgust':'sad',
            'fear': 'sad'
        }
        for m in map_words.keys():
            src_idx = emotion_to_index[m]
            tgt_idx = emotion_to_index[map_words[m]]
            label_array[label_array == src_idx] = tgt_idx
            pred_array[label_array == src_idx] = tgt_idx
        precision = metrics.precision_score(label_array, pred_array, average='micro')
        recall = metrics.recall_score(label_array, pred_array, average='micro')
        f1_score = metrics.f1_score(label_array, pred_array, average='weighted')
        label_array, pred_array = np.array(label_array), np.array(pred_array)
        print('precision', precision)
        print('precision_me', float(np.sum(label_array == pred_array)) / float(len(label_array)))
        print('recall', recall)
        print('f1_score', f1_score)
        labels_write = labels.copy()
        labels_write.remove('good')
        print(classification_report(label_array, pred_array, target_names=labels_write))


