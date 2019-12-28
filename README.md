# Emotion Recognition Framework
本框架为集成多种模态的情绪识别框架，通过从视频中提取声音、图像、文本和姿态等信息进行情绪的识别
框架主要为如下流程
1. 数据预处理：从视频中提取出多种模态信息
2. 模型预测：针对不同的模态，使用不同的模型进行预测
3. 模型融合：根据不同模态的预测结果进行识别结果融合
4. 可视化：将不同模态单独预测的结果和模型融合的结果进行可视化

下面为每个模块的具体使用方法

注意：交互式的模式下不能用服务器，交互式通过打开interactive得到，在视频阶段按s开始录音，讲话结束自动结束录音

## 1 数据构造
### 1.1 数据类型
#### 1.1.1 静态数据
config['data_folder'] 代表了当前使用数据的根目录
config["modalities"]["video"]的state包括可以为true和false
* true代表使用模态名称作为文件夹名称的路径提取数据
* false代表不使用该模态

config["modalities"]中剩余模态的state除了true和false之外多了一种状态
* extract，代表从video中对该模态进行提取

config["labels"]中代表存放数据label的文件，为pkl形式，load的dict中
* key是文件名
* value是[标签]（可能存在多个标签）

例如

```
path 为模态名称
"modalities": {
    "video": {
      "state": "true", //使用path中的路径提取video
    },
    "face": {
      "state": "extract", //从video中提取face
    },
    "audio": {
      "state": "true", //从path中提取audio
    },
    "text": {
      "state": "false", //不使用该模态
    },
}
"labels": {
  "state": "true", //标签
  "path": ""
},
```

注意：在单独提供每种模态数据时，应该保证同一个数据的不同模态以相同的名称命名（后缀可以不相同）

#### 1.1.2 实时数据
在该状态下，不会通过config['data_folder']来访问数据，而是直接从摄像头或者麦克风访问数据
##### 多模态 or 单模态
* 从视频中动态提取，选择模态数据，设置config['modalities']中相应模态的state为true即可使用

### 1.2 特征提取
下面介绍如何从视频中提取多种模态的信息

输出为模态对应的名称文件夹下的文件，输出的dict中格式为{前缀(sample)：[文件名称]}
#### 1.2.1 图像
* 输入：视频（.mp4）
* 输出：多张图片，会在video的命名后加上_{frame_numbers}(.jpg)
* 方法：每n帧获取一帧，基于opencv
* 其中帧的数量由config['modalities']['video']['frames_cut']中的int值决定
##### 1.2.1.1 脸部
* 目前尚未实现脸部对齐，因为在脸部模型中会进行自动对齐
* 目前的输出与图像相同
##### 1.2.1.2 姿态
* 目前尚未实现姿态的提取，因为姿态的信息还没有用到
* 目前的输出与图像相同
#### 1.2.2 声音
* 输入：视频（.mp4）
* 输出：声音（.wav)
* 方法：基于ffmpeg，现在还不能在pycharm上提取成功，只能在命令行里面成功
#### 1.2.3 文本
尚未实现，目前是用直接已经提供好的文本
* 输入：声音
* 输出：文本
* 方法：我猜未来会使用讯飞语音的语音识别接口来进行转换
### 1.3 标签提取
主要包括两个过程
* 第一个是获得前缀名称对应的标签，
* 第二个是将不同的模态数据的标签映射为相同的表达方式

#### 1.3.1 从文件名中提取
此阶段得到的结果是{前缀：label}（label是文本的形式）
* 只需要更改wrapper中get_label的函数，输出前缀和label的dict即可

#### 1.3.2 单词映射
这里会用到util中将所有不同的情绪的单词，转换成同一种单词的方法

## 2 模型预测
### 2.1 audio
* 基于Emotion-Classification-Ravdess
* github连接：https://github.com/marcogdepinto/Emotion-Classification-Ravdess
* 只有pretrained的model，没有代码实现
* pretrained model是基于ravdess训练的

### 2.2 face 
* 基于Emotion-classification-Ravdess
* github连接：https://github.com/sunniee/Emotion_classification_Ravdess
* pretrained model是基于ravdess训练的

### 2.3 text
* 原本用的是基于friends训练的一个文本情感预测模型，但是我测了一下i am happy预测的结果都是错的，所以我就不想用他了
* 目前打算两种思路，一个是百度的API，另一个是模型
* 需要考虑一下我们支持的是中文还是英文
* 还需要考虑从audio到text的转换 
更新1108：
* 目前采用百度的API进行从录音到文本的语音识别，效果还不错
* 但是该模态还不支持从静态的文件中提取，等待实现

### 2.4 pose
待实现

## 3 模型融合
目前没有融合，就把每个结果都输出

## 4 运行方法
按照上述方法更改好config文件后，执行python wrapper.py