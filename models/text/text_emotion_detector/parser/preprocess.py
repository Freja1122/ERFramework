import os
import re
import string
import numpy as np
from zhon.hanzi import punctuation
import pkuseg
import os
import sys
import thulac

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

class TextPreprocess:
    # 用来处理数据的正则表达式
    DIGIT_RE = re.compile(r'\d+')
    LETTER_RE = re.compile(r'[a-zA-Z]+')
    SPECIAL_SYMBOL_RE = re.compile(r'[^\w\s\u4e00-\u9fa5]+')
    NAMED_ENTITY = re.compile(r'[SBIE]+')
    STOPS = ['。', '.', '?', '？', '!', '！']  # 中英文句末字符

    def __init__(self, config={}, verbose=False):
        self._config = config
        self.seg_cixing = thulac.thulac()
        self.seg = pkuseg.pkuseg()
        self.stopwords = TextPreprocess.read_text_file('stopWord.txt')
        self.is_split_sent = False

    @staticmethod
    def read_text_file(text_file):
        stop_file = os.path.join(root_path, 'text_emotion_detector', 'data', text_file)
        """读取文本文件,并返回由每行文本作为元素组成的list."""
        with open(stop_file, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file]
        return lines

    @staticmethod
    def write_text_file(text_list, target_file):
        """将文本列表写入目标文件

        Args:
            text_list: 列表，每个元素是一条文本
            target_file: 字符串，写入目标文件路径
        """
        with open(target_file, 'w', encoding='utf-8') as writer:
            for text in text_list:
                writer.write(text + '\n')

    @staticmethod
    def merge_files(filedir, target_file):
        """
        合并一个文件夹中的文本文件。注意：需要合并的每个文件的结尾要有换行符。

        Args:
            filedir: 需要合并文件的文件夹
            target_file: 合并后的写入的目标文件
        """
        filenames = os.listdir(filedir)
        with open(target_file, 'a', encoding='utf-8') as f:
            for filename in filenames:
                filepath = os.path.join(filedir, filename)
                f.writelines(open(filepath, encoding='utf-8').readlines())

    @staticmethod
    def del_blank_lines(sentences):
        """删除句子列表中的空行，返回没有空行的句子列表

        Args:
            sentences: 字符串列表
        """
        sentences = [s for s in sentences if s]
        return sentences

    @staticmethod
    def del_punctuation(sentence):
        """删除字符串中的中英文标点.

        Args:
            sentence: 字符串
        """
        en_punc_tab = str.maketrans('', '', string.punctuation)  # ↓ ① ℃处理不了
        sent_no_en_punc = sentence.translate(en_punc_tab)
        return re.sub(r'[%s]+' % punctuation, "", sent_no_en_punc)

    def del_stopwords(self, seg_sents, cixing=False):
        """删除句子中的停用词

        Args:
            seg_sents: 嵌套列表，分好词的句子（列表）的列表
            stopwords: 停用词列表

        Returns: 去除了停用词的句子的列表
        """
        if self.is_split_sent:
            seg_sents = [[word for word in sent if word not in self.stopwords] for sent in seg_sents]
        else:
            if cixing:
                seg_sents = [sent for sent in seg_sents if sent[0] not in self.stopwords]
            else:
                seg_sents = [sent for sent in seg_sents if sent not in self.stopwords]
        return seg_sents

    def cut_sent(self, para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")


    def del_special_symbol(cls, sentence):
        """删除句子中的乱码和一些特殊符号。"""
        return cls.SPECIAL_SYMBOL_RE.sub('', sentence)

    def del_english_word(cls, sentence):
        """删除句子中的英文字符"""
        return cls.LETTER_RE.sub('', sentence)

    def seg_sentences(self, sentences):
        """对输入的字符串列表进行分词处理,返回分词后的字符串列表."""
        if isinstance(sentences,str):
            text_cixing = self.seg_cixing.cut(sentences)
        elif isinstance(sentences,list):
            text_cixing = [self.seg_cixing.cut(s) for s in sentences]
        return text_cixing

    def preprocess(self, sentences):
        # # 分句
        if self.is_split_sent:
            sentences = self.cut_sent(sentences)
        # 分词
        sentences_cixing = self.seg_sentences(sentences)
        # 去停用词
        sentences_cixing = self.del_stopwords(sentences_cixing, cixing=True)
        # sentences = TextPreprocess.del_blank_lines(sentences)
        return sentences_cixing



if __name__ == "__main__":
    text_preprocess = TextPreprocess()
    res = text_preprocess.preprocess('我好难啊')
    print(res)