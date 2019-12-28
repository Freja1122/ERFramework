import csv


class EmotionDictionary:
    emotion_map = {
        'neutral': ['PD'],
        'happy': ['PA', 'PE'],
        'good': ['PH', 'PG', 'PB', 'PK'],
        'angry': ['NA'],
        'sad': ['NB', 'NJ', 'NH', 'PF', 'NG'],
        'fear': ['NI', 'NC'],
        'disgust': ['NE', 'ND', 'NN', 'NK', 'NL'],
        'surprise': ['PC']
    }

    def __init__(self):
        self.reverse_emotion = {
            'happy': ['sad'],
            'good': ['sad'],
            'angry': ['neutral'],
            'sad': ['neutral'],
            'fear': ['neutral'],
            'disgust': ['neutral'],
            'surprise': ['neutral'],
            'neutral': ['neutral'],
        }
        # 组织成 '单词': [一级分类，二级分类]的形式
        emotion_map_reverse = {vv: k for k, v in list(self.emotion_map.items()) for vv in v}

        # 读取csv至字典
        csvFile = open("../data/emotion_dictionary/emotion_dictionary.csv", "r")
        reader = csv.reader(csvFile)

        # 建立字典
        result = {}
        for item in reader:
            if item[0] in result:
                continue
            # 忽略第一行
            if reader.line_num == 1:
                continue
            emotion = item[4].strip()
            result[item[0]] = {'emotion': [emotion_map_reverse[emotion], emotion],
                               'extend': int(item[5].strip())}

        csvFile.close()
        self.dictionary = result


if __name__ == "__main__":
    print(EmotionDictionary().dictionary['开心'])
