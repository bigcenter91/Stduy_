# 27_1번을 for문으로 써서 바꿔
# xgboost 메시지 지우고


import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# 피쳐 = 컬럼, 열, 특성
# 트리계열에서는 이상치를 제거하지 않아도 된다// nan이나 결측치가 있어도 돌아가
# 스케일링을 하지 않아도 된다

# 트리계열 처음에 떠오르는 것 DecisionTreeClassifier

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
model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

#3. 훈련
model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)


import matplotlib.pyplot as plt

def plot_feature_importances(model) :
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.title(model)
    

model_list = [DecisionTreeClassifier(), RandomForestClassifier(), 
              GradientBoostingClassifier(), XGBClassifier()]

for i in range(4):
    
    globals()['model'+str(i)] = model_list[i]
    globals()['model'+str(i)].fit(x_train, y_train)
    plt.subplot(2,2, i+1)
    print(globals()['model'+str(i)].feature_importances_)
    plot_feature_importances(globals()['model'+str(i)])
    if i ==3:
        plt.title('XGBClassifier()')
plt.show()
