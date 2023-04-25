# [실습]
# 피쳐임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성 후
# 모델을 돌려서 결과 도출

# 기존 모델들과 성능비교
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score



#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

allfeature = round(x.shape[1]*0.2, 0)
print('자를 갯수: ', int(allfeature))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle=True, random_state=1234 
) 


# def plot_feature_importances(model) :
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Feature')
#     plt.ylim(-1, n_features)
#     plt.title(model)
    

#2. 모델구성

models = [DecisionTreeClassifier(),RandomForestClassifier(),GradientBoostingClassifier(),XGBClassifier()]


# for i in range(len(models)) :
#     model = models[i]
#     name = str(model).strip('()')
#     model.fit(x_train, y_train)
#     result = model.score(x_test, y_test)
#     fimp = model.feature_importances_
#     print("="*100)
#     print(name,'의 결과값 : ', result)
#     print('model.feature_importances : ', fimp)
#     print("="*100)  
#     #plt.subplot(2, 2, i+1)
#     #plot_feature_importances(models[i])#
# #    if str(models[i]).startswith("XGB") : 
# #        plt.title('XGBRegressor')
# #    else :
# #        plt.title(models[i])

# #plt.show()


for model in models:
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    if str(model).startswith('XGB'):
        print('XGB 의 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 스코어: ', score)
        
    featurelist = []
    for a in range(int(allfeature)):
        featurelist.append(np.argsort(model.feature_importances_)[a])
        
    x_bf = np.delete(x, featurelist, axis=1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x_bf, y, shuffle=True, train_size=0.8, random_state=1234)
    model.fit(x_train2, y_train2)
    score = model.score(x_test2, y_test2)
    if str(model).startswith('XGB'):
        print('XGB 의 드랍후 스코어: ', score)
    else:
        print(str(model).strip('()'), '의 드랍후 스코어: ', score)



# 결과 비교
# 예)
# 1. DecisionTree
# 기존 acc : 
# 컬럼 삭제 후 acc : 

# 2. RandomForest
# 기존 acc : 
# 컬럼 삭제 후 acc : 

# 3. GradientDecentBoosting
# 기존 acc : 
# 컬럼 삭제 후 acc : 

# 4. XGBoost
# 기존 acc : 
# 컬럼 삭제 후 acc : 

