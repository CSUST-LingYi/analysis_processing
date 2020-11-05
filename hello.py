# coding:utf8
from flask import Flask, make_response, jsonify, request
import pandas as pd
import numpy as np
import cpca
import re
import datetime
from os import path, remove, listdir
import chnSegment
import plotWordcloud
import scoreTraining
import loadScoreModel
import scoreSignalPredict

app = Flask(__name__)

resp = {
        "code": 201,
        "msg": "data_processing_completed",
        "createTime": datetime.datetime.now()
}


# 学生地址信息处理
def stu_map_analysis(strlist):
    print("开始清洗地址数据==============>>>>>>")
    # json数组字符串解构
    address = strlist.get('data')
    # before_split = list[1:len(list)-1]
    # address = before_split.split(',')
    c = {"address": address}
    df = pd.DataFrame(c, columns=['address'])
    adlist = df['address'].tolist()
    # 地址分词
    cut = cpca.transform(adlist)
    # 替换表中的空字符串为NaN
    cut.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    # 删除'省'有空值的行，在原表更改
    cut.dropna(subset=['省'], inplace=True)
    provincelist = cut['省'].tolist()
    mplist = []
    for i in range(len(provincelist)):
        s = provincelist[i]
        # 删除两端空白
        strips = re.sub('D', '', s)
        # 删除后缀名
        res = re.search('(?P<province>[^省|市|壮族自治区|自治区|维吾尔自治区|回族自治区|特别行政区]+)', strips)
        mplist.append(res.groupdict().get('province'))
    mplist = pd.DataFrame(mplist)
    counts = mplist[0].value_counts()
    data = []
    # 组装响应数据
    for i in range(len(counts)):
        data.append({
            "name": counts.index[i],
            "value": counts[i]
        })
    print("学生地址信息清理完成============>>>>>>")
    return data


# 生成学生词云图
def stu_word_cloud_create(req):
    text = ','.join(req.get("data"))
    # 分词
    try:
        text = chnSegment.word_segment(text)
        # 生成词云
        image_path = plotWordcloud.generate_wordcloud(text)
    except ValueError:
        print("词云处理错误")
        return "0"
    else:
        return image_path


@app.route("/")
def index():
    return "<h1 style='text-align:center;'>当前资源不支持浏览器访问！</h1>"


# 生源地图
@app.route("/addressMap", methods=['POST'])
def address_map():
    req = request.get_json()
    data = stu_map_analysis(request.get_json())

    if data is None:
        resp.update({
            'code': 404,
            'msg': 'RESOURCES_NOT_FOUND'
        })
        recd = make_response(jsonify(resp), 404)
        return recd
    resp["data"] = [str(i) for i in data]
    # resp["data"] = data
    recd = make_response(jsonify(resp), 200)
    return recd


# 学生词云图
@app.route("/stuwordcloud", methods=['POST'])
def stu_word_cloud():
    req = request.get_json()
    img_path = stu_word_cloud_create(req)
    if img_path == "0":
        resp.update({
            "code": 500,
            "msg": "CREATE FAILED",
        })
        return make_response(jsonify(resp), 500)
    resp["data"] = img_path
    return make_response(jsonify(resp), 200)


# 清理词云图图片
@app.route("/clearpicbuffer", methods=['POST'])
def del_file():
    req = request.get_json()
    if (req.get("token") != "lingyi") | (req is None):
        return make_response("PERMISSION_DENIED", 401)
    d = path.dirname(__file__)
    wholepath = path.join(d, 'static/Images')
    try:
        ls = listdir(wholepath)
        for i in ls:
            c_path = path.join(wholepath, i)
            if path.isdir(c_path):
                del_file(c_path)
            else:
                remove(c_path)
    except FileNotFoundError:
        return make_response("DELETE_FAILED_NOT_SUCH_FILE", 404)
    else:
        return make_response("DELETE_SUCCESS", 200)


# 添加用户自定义词典
@app.route("/adduserdict", methods=['POST'])
def add_user_dict():
    req = request.get_json()
    req = req.get("data")
    if chnSegment.add_userdict_as_list(req):
        resp["data"] = "字典添加成功"
        return make_response(jsonify(resp), 201)
    resp.update({
        "code": 500,
        "msg": "CREATE FAILED",
    })
    return make_response(jsonify(resp), 500)


# 添加停用词
@app.route("/addstopwords", methods=['POST'])
def add_stop_words():
    req = request.get_json()
    req = req.get("data")
    if plotWordcloud.add_stopwords(req):
        resp["data"] = "停用词添加成功"
        return make_response(jsonify(resp), 201)
    resp.update({
        "code": 500,
        "msg": "CREATE FAILED",
    })
    return make_response(jsonify(resp), 500)


# 聚类训练
@app.route("/scoreTrain",methods=['POST'])
def score_training():
    req = request.get_json()
    req = req.get("data")
    if req != "lingyi":
        resp.update({
            "code": 403,
            "msg": "NO PERMISSION"
        })
        resp['data'] = 'ERROR'
        return make_response(jsonify(resp), 403)
    else:
        scoreTraining.scoreTraining()


# 分数画像预测
@app.route("/scoreSignal", methods=['POST'])
def score_signal():
    '''
        参数数组：8个参数，依次为平均分，标准差，最高分，最低分，<60课程数，60-80课程数，80-90课程数，>90课程数
    '''
    req = request.get_json()
    req = req.get("data")
    result = loadScoreModel.predict_cluster(req)
    if any(result) == -1:
        resp.update({
            "code": 400,
            "msg": "VALUE ERROR"
        })
        resp['data'] = None
        return make_response(jsonify(resp), 400)
    resp.update({
        "code": 201,
        "msg": "SUCCESS"
    })
    resp['data'] = result.tolist()
    return make_response(jsonify(resp), 201)


if __name__ == "__main__":
   # app.run("192.168.184.1")
   #  req = {'data': ['优秀团员', 't', '全国大学生英语竞赛', '高等数学C（二）', '信息管理与信息系统专业导论', '大学生学习方法指导', '中国近现代史纲要', '计算机系统与系统软件', '湖南省大学生电子商务大赛二等奖', '长沙理工大学第二届“创新项目策划大赛”二等奖', '长沙理工大学2019年“挑战杯”大学生课外学术科技作品竞赛二等奖', '女生委员', '长沙理工大学第五届“互联网+”大学生创新创业大赛三等奖', '长沙理工大学“第十三届企业模拟经营大赛”一等奖', '长沙理工大学英语阅读大赛三等奖', '优秀寝室', '第三届全国大学生预防艾滋病知识竞赛', '暑假优秀班级', '班服比赛二等奖', '爱可秀', '迟到', '无']}
   #  # print(stu_word_freq(req))
   #         scoreTraining.scoreTraining()
   #  scoreSignalPredict.predictTraining("./doc/score_cluster_res.csv")
           print(loadScoreModel.predict_cluster([[83.7333,7.370813312578801,95,71,0,0,5,8,3]]))
   #  import pymysql
   #  con = pymysql.connect(host='localhost', user='python', password='python', database='test2020', charset='utf8',
   #                  use_unicode=True)
   #  sql_cmd = 'select * from view_scoreanalyses'
   #  df = pd.read_sql(sql_cmd, con)
   #  print(df.head())