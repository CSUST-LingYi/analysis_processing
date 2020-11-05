# coding:utf-8

from collections import Counter
from os import path
import jieba
jieba.load_userdict(path.join(path.dirname(__file__),'userdict//userdict.txt')) # 导入用户自定义词典


def word_segment(text):
    '''
    通过jieba进行分词并通过空格分隔,返回分词后的结果
    '''

    # 计算每个词出现的频率，并存入txt文件
    # jieba_word=jieba.cut(text,cut_all=False) # cut_all是分词模式，True是全模式，False是精准模式，默认False
    # data=[]
    # for word in jieba_word:
    #     data.append(word)
    # dataDict=Counter(data)
    # with open('doc//词频统计.txt', 'w') as fw:
    #     for k,v in dataDict.items():
    #         fw.write("%s,%d\n" % (k,v))
        #  fw.write("%s"%dataDict)

    # 返回分词后的结果
    jieba_word=jieba.cut(text,cut_all=False) # cut_all是分词模式，True是全模式，False是精准模式，默认False
    seg_list=' '.join(jieba_word)
    return seg_list


# 批量添加用户自定义字典
def add_userdict_as_list(word_list):
    try:
        with open('userdict//userdict.txt', 'a', encoding='utf8') as fw:
            for i in range(len(word_list)):
                fw.write("%s\n" % (word_list[i],))
    except IOError:
        print("用户字典添加错误");
        return False
    else:
        fw.close()
        return True

