#### 분류데이터들 싹모아서 테스트#####
'''
import numpy as np
from sklearn.datasets import load_iris,load_diabetes,load_breast_cancer,load_digits,fetch_covtype,load_wine

1. data
datasets = load_iris()
x = datasets.data
y = datasets['target']
x,y = load_iris(return_X_y=True)
x,y = load_diabetes(return_X_y=True)
x,y = load_breast_cancer(return_X_y=True)
x,y = load_digits(return_X_y=True)
x,y = fetch_covtype(return_X_y=True)
x,y = load_wine(return_X_y=True)

#print(x.shape,y.shape) #(150, 4) (150,)

#2. model

from sklearn.svm import LinearSVC # 이 모델로 리폼.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #분류모델 DecisionTreeRegressor 회귀모델
from sklearn.ensemble import RandomForestRegressor
#tree모델 randomforest 는 숲이여서 나무들이 모여있음
#model =LinearSVC(C=10) #-> 알고리즘연산이 다 포함되어있음. = 비지도학습의 kmean과 비슷.
#model = LogisticRegression() # -> 얘는 원래 이진분류
#model = DecisionTreeRegressor() #-> 원래 안되는게 정상
#model = DecisionTreeClassifier() #분류모델
model = RandomForestRegressor() #0.992474

#3. compile, fit

model.fit(x,y)

#4. evaluate , predict

results = model.score(x,y)

print(results) # 딥러닝[0.06923310458660126, 0.9800000190734863] 머신러닝 #0.9666666666666667

#머신러닝안에 딥러닝이 들어감.
'''
import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
# Define a list of models
models = [LinearSVC(max_iter=10000), RandomForestClassifier()]

# Loop through each dataset and model, fit the model and print the score

datasets = [(load_iris(), 'Iris'), 
            (load_breast_cancer(), 'Breast Cancer'), 
            (load_digits(), 'Digits'), 
            (fetch_covtype(), 'Covtype'), 
            (load_wine(), 'Wine')]

scaler = RobustScaler()

for dataset, name in datasets:
    x, y = dataset['data'], dataset['target']  # unpack dataset object from tuple
    x = scaler.fit_transform(x)
    for model in models:
        model.fit(x, y)
        score = model.score(x, y)
        print(f"Dataset: {name}, Model: {type(model).__name__}, Score: {score}")
# Dataset: Iris, Model: LinearSVC, Score: 0.9466666666666667
# Dataset: Iris, Model: RandomForestClassifier, Score: 1.0
# Dataset: Breast Cancer, Model: LinearSVC, Score: 0.9876977152899824
# Dataset: Breast Cancer, Model: RandomForestClassifier, Score: 1.0
# Dataset: Digits, Model: LinearSVC, Score: 0.9938786867000556
# Dataset: Digits, Model: RandomForestClassifier, Score: 1.0
# Dataset: Covtype, Model: LinearSVC, Score: 0.7127150557991918
# Dataset: Covtype, Model: RandomForestClassifier, Score: 1.0
# Dataset: Wine, Model: LinearSVC, Score: 1.0
# Dataset: Wine, Model: RandomForestClassifier, Score: 1.0