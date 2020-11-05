from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def predictTraining(filename):
    al = pd.read_csv(filename)
    # weight = al[['avgScore','stdScore','maxScore','minScore','lt60','gt60lt80','gt80lt90','gt90']]
    weight = al[['avgScore', 'stdScore', 'maxScore', 'minScore', 'lt60', 'gt60lt70', 'gt70lt80', 'gt80lt90', 'gt90']]
    labels = al['cluster']

    x_train, x_test, y_train, y_test = train_test_split(weight, labels, test_size=0.2, random_state=14)

    RF_model = ExtraTreesClassifier(n_estimators=111)
    RF_model.fit(x_train, y_train)
    print("极端随机树预测准确率：", RF_model.score(x_test, y_test))

    '''
        8、保存模型
    '''
    joblib.dump(RF_model, './model/score_RF_model.pkl')