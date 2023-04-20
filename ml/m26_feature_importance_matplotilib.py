import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
# 피쳐 = 컬럼, 열, 특성
# 트리계열에서는 이상치를 제거하지 않아도 된다// nan이나 결측치가 있어도 돌아가
# 스케일링을 하지 않아도 된다

#1. 데이터
datasets = load_iris() # 컬럼 4개
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=337, shuffle=True
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier()
# 위에 공통점이 트리 계열이라는거다


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("=================================")
print("accuracy_score : ", acc)
print(model, ":", model.feature_importances_)

import matplotlib.pyplot as plt

def plot_feature_importances(model) :
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.title(model)
    
plot_feature_importances(model)
plt.show()

# 트리계열에서만 제공해준다 피쳐_임포턴스

# accuracy_score : 0.9333333333333333
# DecisionTreeClassifier() : [0.02506789 0.         0.54934776 0.42558435]

# accuracy_score :  0.9666666666666667
# RandomForestClassifier() : [0.12127547 0.02842019 0.40685523 0.44344911]

# accuracy_score :  0.9666666666666667
# GradientBoostingClassifier() : [0.00311363 0.01628157 0.7981164  0.1824884 ]

# accuracy_score : 0.9666666666666667
# XGBClassifier : [0.01794496 0.01218657 0.8486943  0.12117416] 


'''
#1. 데이터
data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_wine(return_X_y=True),
             load_digits(return_X_y=True)]

data_name_list = ['아이리스 : ',
                  '캔서 : ',
                  '와인 : ',
                  '디지트 : ']

model_list = [DecisionTreeClassifier(),
              RandomForestClassifier(),
              GradientBoostingClassifier(),
              XGBClassifier()]

model_name_list = ['DecisionTreeClassifier : ',
                  'RandomForestClassifier : ',
                  'GradientBoostingClassifier : ',
                  'XGBClassifier : ']

#2. 모델
for i, v in enumerate(data_list):
    x, y = v
    print("===================================")
    print(data_name_list[i])
    for j, v2 in enumerate(model_list):
            model = v2
                        
            #3. 컴파일, 훈련
            model.fit(x_train, y_train)
            
            #4. 평가, 예측
            results = model.score(x_test, y_test)
            print(model_name_list[j], results)
            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test, y_predict)
            print(model_name_list[j], "accuracy_score:", acc)
            
            if i !=3:
                print(model, ":", model.feature_importances_)
            else
            
            #5. feature importance 출력
            print(model, ":", model.feature_importances_)
            '''