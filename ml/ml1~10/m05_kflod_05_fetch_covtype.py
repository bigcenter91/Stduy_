import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings(action = 'ignore')

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             fetch_covtype(return_X_y=True),
             load_wine(return_X_y=True)]

model_list = [LinearSVC(),
              LogisticRegression(),
              DecisionTreeClassifier(),
              RandomForestClassifier()]

data_name_list = ['iris : ',
                  'breast_cancer :',
                  'digits :',
                  'covtype',
                  'load_wine']

model_name_list = ['LinearSVC',
              'LogisticRegression',
              'DecisionTreeClassifier',
              'RandomForestClassifier']

n_splits=5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 413)

#2. 모델
for  i, value in enumerate(data_list): #enumerate수치와 순서를 나타내주는 함수.
    x, y = value #첫번째 iris들어가고 두번째 cancer가 들어감
    #print(x.shape,y.shape)
    print("=============================")
    print(data_name_list[i])
    
    for j, value2 in enumerate(model_list):
        model = value2 #j가 들어가면 데이터값이 나와버림
        #3. 컴파일, 훈련
        scores = cross_val_score(model, x, y, cv = kfold)
        #4. 평가, 예측
        print('ACC :', scores, 
      '\n cross_val_score average : ', round(np.mean(scores),4))
        
# covtype
# ACC : [0.63459635 0.50435875 0.62474828 0.63562589 0.6327688 ]
#  cross_val_score average :  0.6064
# ACC : [0.6199926  0.62298736 0.62171047 0.61918039 0.61831122]
#  cross_val_score average :  0.6204
# ACC : [0.9403888  0.93895166 0.93838316 0.94002685 0.9385983 ]
#  cross_val_score average :  0.9393
# ACC : [0.95584451 0.95558634 0.95420905 0.95562039 0.95462212]
#  cross_val_score average :  0.9552