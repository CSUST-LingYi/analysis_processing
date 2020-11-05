# coding:utf-8

from os import path
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
import uuid
from wordcloud import WordCloud


def generate_wordcloud(text):
    '''
    输入文本生成词云,如果是中文文本需要先进行分词处理
    '''
    # 设置显示方式
    d=path.dirname(__file__)
    # mask = np.array(Image.open(path.join(d, "maskpic//mask.png")))
    font_path=path.join(d,"font//msyh.ttf")
    stopwords = [line.strip() for line in open("userdict/stopword.txt", 'r', encoding='utf-8').readlines()]
    stopwords = set(stopwords)
    wc = WordCloud(background_color="white",# 设置背景颜色
           max_words=2000, # 词云显示的最大词数
           stopwords=stopwords, # 设置停用词
           font_path=font_path, # 兼容中文字体，不然中文会显示乱码
           width=500,
           height=350
             )

    # 生成词云 
    wc.generate(text)
    uuid_str = uuid.uuid4().hex
    # 生成的词云图像保存到本地
    image_path = r"static/Images/"+uuid_str+".png"
    wc.to_file(path.join(d, image_path))

    # 显示图像
    # plt.imshow(wc, interpolation='bilinear')
    # # interpolation='bilinear' 表示插值方法为双线性插值
    # plt.axis("off")# 关掉图像的坐标
    # plt.show()
    return image_path

# 添加停用词库
def add_stopwords(sw_list):
    try:
        with open('userdict//stopword.txt', 'a', encoding='utf8') as fw:
            # fw.write("\n")
            for i in range(len(sw_list)):
                fw.write("%s\n" % (sw_list[i]))
    except IOError:
        print("停用词添加错误")
        return False
    else:
        fw.close()
        return True
