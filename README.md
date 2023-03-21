# sentiment_classification

英文情感分类.

交互的demo部署到了huggingface:
https://huggingface.co/spaces/szzzzz/sentiment_classification

## 安装

```shell
git clone git@github.com:szprob/sentiment_classification.git
cd sentiment_classification
python setup.py install --user
```

或者直接

```shell
pip install git+https://github.com/szprob/sentiment_classification.git
```

## 模型

模型backbone是roberta.

分类头有三个,分别是2分类(1:正向,0:负向),3分类(2:正,1:中,0:负),5分类(0-4分别对应1星-5星).


预训练模型全部开源,可以直接下载,也可以直接在代码中读取远端模型.

16m参数模型:

百度云地址：链接：https://pan.baidu.com/s/1tzcY98JuQ75XoPzjzLQGnA 提取码：qewg

huggingface : https://huggingface.co/szzzzz/sentiment_classifier_sentence_level_bert_16m


## 数据集

本项目使用的数据集是多个情感分类数据集的结合.包括不限于:

1. imdb. 2分类数据集,可以用huggingface的datasets库读取: datasets.load_dataset("imdb")

2. rotten_tomatoes. 2分类数据集,同样: datasets.load_dataset("rotten_tomatoes")

3. tweet_eval. 3分类数据集,datasets.load_dataset("tweet_eval", name="sentiment")

4. yelp_review_full. 5分类数据集, datasets.load_dataset("yelp_review_full")

5. steam_reviews. 2分类数据集

6. Video_Games_5. 5分类数据集


## 数据处理


1. 实际训练中可以适当串联一下多个分类头任务数据.

比如5分类任务的1星和2星,可以当做3分类的负面评论,辅助训练三分类任务头.

2. 表情数据可以使用emoji库做文本化.

3. 无需做标点等停用词去除

4. 过长语句做截断



## 有序分类

情感分析是一个有序分类任务,可以有多种方法做处理,如.

1. 在普通的CELoss上加入权重,权重可以是 |y_pred_label-y_true_label]/tag_num

2. 将有序分类问题转移为回归问题. 标签如[0,1,2,3],可以转化为[0.125, 0.375, 0.625, 0.875].

3. 当作多标签的2分类任务,如1星label可以标做[0,0,0,0],2星label标做[1,0,0,0],以此类推

本质上,就是要在分类loss里引入回归的思想,也就是说在真实标签和预测标签的类别之间比较远时,增加loss,反之减少loss.

本项目使用的是第三种,loss使用的是coral loss,详情可见:https://github.com/Raschka-research-group/coral-pytorch



## 使用

文本目前只做了英文,只使用了5个类别的分类头.

使用方法如下:

```python
from sentiment_classification import TextToxicDetector

model = TextToxicDetector()

# 如果模型down到了本地
model.load(model_path)
# 也可以直接使用远端
model.load('szzzzz/sentiment_classifier_sentence_level_bert_16m')

# 模型预测
result = model.rank("I like it.")
'''
result
{'toxic': 0.94,
 'severe_toxic': 0.03,
 'obscene': 0.59,
 'threat': 0.02,
 'insult': 0.44,
 'identity_hate': 0.05}
'''

```
