import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from sklearn import preprocessing
import joblib
import pymysql


def scoreTraining():
    try:
        con = pymysql.connect(host='localhost', user='python', password='python', database='test2020', charset='utf8',
                            use_unicode=True)
        sql_cmd = 'select * from view_scoreanalyses'
        sc = pd.read_sql(sql_cmd, con)
    except ConnectionError:
        return "连接数据库失败"
    '''
        添加极端数据进行训练
    '''
    min_adjust = {
        "studentNo": "201844070000",
        "termYear": 2018,
        "xuenian": "2018-2019",
        "avgScore": 40,
        "stdScore": 0,
        "maxScore": 40,
        "minScore": 40,
        "fails": "",
        "lt60": 12,
        "gt60lt70": 0,
        "gt70lt80":0,
        "gt80lt90": 0,
        "gt90": 0
    }
    max_adjust = {
        "studentNo": "201744070000",
        "termYear": "2018",
        "xuenian": "2018-2019",
        "avgScore": 95,
        "stdScore": 0,
        "maxScore": 95,
        "minScore": 95,
        "fails": "",
        "lt60": 0,
        "gt60lt70": 0,
        "gt70lt80": 0,
        "gt80lt90": 0,
        "gt90": 12
    }
    sc = sc.append(min_adjust,ignore_index=True)
    sc = sc.append(max_adjust,ignore_index=True)
    sc1 = sc.copy()
    sc['avgScore']= preprocessing.scale(sc['avgScore'])
    sc['stdScore']= preprocessing.scale(sc['stdScore'])
    sc['maxScore']= preprocessing.scale(sc['maxScore'])
    sc['minScore']= preprocessing.scale(sc['minScore'])
    sc['lt60'] = preprocessing.scale(sc['lt60'])
    sc['gt60lt70'] = preprocessing.scale(sc['gt60lt70'])
    sc['gt70lt80'] = preprocessing.scale(sc['gt70lt80'])
    sc['gt80lt90'] = preprocessing.scale(sc['gt80lt90'])
    sc['gt90'] = preprocessing.scale(sc['gt90'])
    print(sc.tail())

    # sc = pd.read_csv(trainFilePath)
    sc.fillna('无', inplace=True)
    ts1 = sc[['avgScore', 'stdScore', 'maxScore', 'minScore','lt60','gt60lt70','gt70lt80','gt80lt90','gt90']]

    outfile = r'./output.xls'  # 设置输出文件的位置
    '''
        聚类
    '''
    #数据标准化
    ts1 = pd.DataFrame(preprocessing.scale(ts1))
    # K参数调优并可视化
    SSE = []
    for k in range(2,10):
        km = KMeans(n_clusters=k)
        s = km.fit(ts1)
        num = len(ts1)
        center = km.labels_
        SSE.append(km.inertia_)
    #     print(center,type(center))
    #     print(k)
    #     print('聚类效果：',km.inertia_)
        print(k,silhouette_score(ts1,center, metric='euclidean'))

    # 可视化效果
    X = range(2, 10)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    # plt.show()
    plt.savefig("./img/diff_k.png")
    plt.close()

    # k-means 训练模型
    mod = KMeans(n_clusters=4, n_jobs=4, max_iter=1800, random_state=14)  # 聚成4类数据,并发数为4，最大循环次数为1800
    mod.fit_predict(ts1)  # y_pred表示聚类的结果
    # 轮廓系数
    print("轮廓系数:", silhouette_score(ts1, mod.predict(ts1)))

    # 聚成4类数据，统计每个聚类下的数据量，并且求出他们的中心
    r1 = pd.Series(mod.labels_).value_counts()
    r2 = pd.DataFrame(mod.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(ts1.columns) + [u'类别数目']
    r = pd.concat([ts1, pd.Series(mod.labels_, index=ts1.index)], axis=1)
    # 给每一条数据标注上被分为哪一类

    r.columns = list(ts1.columns) + [u'cluster']
    print(r.head())

    # h.to_excel(outfile)#如果需要保存到本地，就写上这一列
    print("聚类中心：\n", mod.cluster_centers_)

    # 聚类可视化过程

    tsne = TSNE()

    # 绘制散点图
    tsne.fit_transform(r)
    tsne = pd.DataFrame(tsne.embedding_, index=r.index)

    a = tsne[r[u'cluster'] == 0]
    plt.plot(a[0], a[1], 'r.')
    a = tsne[r[u'cluster'] == 1]
    plt.plot(a[0], a[1], 'go')
    a = tsne[r[u'cluster'] == 2]
    plt.plot(a[0], a[1], 'b*')
    a = tsne[r[u'cluster'] == 3]
    plt.plot(a[0], a[1], 'y*')
    a = tsne[r[u'cluster'] == 4]
    plt.plot(a[0], a[1], 'k+')
    a = tsne[r[u'cluster'] == 5]
    plt.plot(a[0], a[1], 'm*')
    a = tsne[r[u'cluster'] == 6]
    plt.plot(a[0], a[1], 'c.')
    a = tsne[r[u'cluster'] == 7]
    plt.plot(a[0], a[1], c='pink')

    plt.savefig("./img/cluster.png")

    sc1['cluster'] = r['cluster']
       # r['studentNo'] = sc['studentNo']
    sc1.loc[sc1['cluster'] == 1, 'cluster'] = 'blue'
    sc1.loc[sc1['cluster'] == 2, 'cluster'] = 'red'
    sc1.loc[sc1['cluster'] == 3, 'cluster'] = 'green'
    sc1.loc[sc1['cluster'] == 0, 'cluster'] = 'yellow'
    print(sc1.groupby('cluster').mean())
    score_cluster_res_filename = "./doc/score_cluster_res.csv"
    sc1.to_csv(score_cluster_res_filename)

    '''
        分类预测
    '''
    import scoreSignalPredict
    scoreSignalPredict.predictTraining(score_cluster_res_filename)
