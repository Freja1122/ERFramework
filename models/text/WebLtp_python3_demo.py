#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import urllib.request
import urllib.parse
import json
import hashlib
import base64
import ast
from models.text.textPredictions import *

# 接口地址
url = "https://ltpapi.xfyun.cn/v2/sa"
# 开放平台应用ID
x_appid = "5dc44548"
# 开放平台应用接口秘钥
api_key = "83b807bd3516828f6de4faf26dea528d"


# 语言文本


def main(TEXT):
    body = urllib.parse.urlencode({'text': TEXT}).encode('utf-8')
    param = {"type": "dependent"}
    x_param = base64.b64encode(json.dumps(param).replace(' ', '').encode('utf-8'))
    x_time = str(int(time.time()))
    x_checksum = hashlib.md5(api_key.encode('utf-8') + str(x_time).encode('utf-8') + x_param).hexdigest()
    x_header = {'X-Appid': x_appid,
                'X-CurTime': x_time,
                'X-Param': x_param,
                'X-CheckSum': x_checksum}
    req = urllib.request.Request(url, body, x_header)
    result = urllib.request.urlopen(req)
    result = result.read()
    print(result.decode('utf-8'))
    return ast.literal_eval(result.decode('utf-8'))


def get_same_res(file1, file2):
    with open(file1) as f1:
        lines1 = f1.readlines()
    with open(file2) as f2:
        lines2 = f2.readlines()
    for l1 in lines1:
        if l1 in lines2:
            print(l1)


map_2_idx = {
    'positive': 1,
    'negative': -1,
    'neutral': 0
}

if __name__ == '__main__':
    get_same_res('bad_res_xunfei.txt', 'bad_res_baidu.txt')
    # pass
    # lines = get_test_data('data/test_conversation.txt')
    # failed_samples = []
    # result_all = [{}, {}]
    # texts = []
    # for i, l in enumerate(lines):
    #     text = l[0]
    #     print(i)
    #     print(text)
    #     label_1 = l[1]
    #     label_2 = l[2]
    #     res = main(text)
    #     if res.get('desc') == "success":
    #         sentiment = res['data']['sentiment']
    #         score = res['data']['score']
    #         result_all[sentiment == map_2_idx[label_2]][text] = {
    #             'sentiment': sentiment,
    #             'score': score,
    #             'labels': [label_1, label_2]
    #         }
    #     else:
    #         failed_samples.append(l[0])
    # print(result_all)
    # print(len(result_all[0]))
    # print(len(result_all[1]))
    # print(failed_samples)
    # with open('bad_res_xunfei.txt', 'w') as f:
    #     f.write('\n'.join(list(result_all[0].keys())))
    # print('write')
