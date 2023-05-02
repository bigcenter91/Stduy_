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
        scores = cross_val_score(model, x, y, cv = kfold, n_jobs= 4) # n_jobs core가 4개
        #4. 평가, 예측
        print('ACC :', scores, 
      '\n cross_val_score average : ', round(np.mean(scores),4))
        
# breast_cancer :
# ACC : [0.94736842 0.94736842 0.93859649 0.87719298 0.94690265] 
#  cross_val_score average :  0.9315
# ACC : [0.95614035 0.96491228 0.94736842 0.86842105 0.96460177] 
#  cross_val_score average :  0.9403
# ACC : [0.94736842 0.93859649 0.92105263 0.87719298 0.9380531 ] 
#  cross_val_score average :  0.9245
# ACC : [0.96491228 0.97368421 0.94736842 0.93859649 0.95575221] 
#  cross_val_score average :  0.9561