import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,fetch_covtype,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import r2_score, accuracy_score 
warnings.filterwarnings(action= 'ignore') #경고 무시

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             fetch_covtype(return_X_y=True),
             load_wine(return_X_y=True)] #->매우중요!!!!!!!!! for문을 쓰기위해 list형태로 만들어 놓음. 

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
#2. 모델
for  i, value in enumerate(data_list): #enumerate수치와 순서를 나타내주는 함수.
    x, y = value #첫번째 iris들어가고 두번째 cancer가 들어감
    #print(x.shape,y.shape)
    print("=============================")
    print(data_name_list[i])
    
    for j, value2 in enumerate(model_list):
        model = value2 #j가 들어가면 데이터값이 나와버림
        #3. 컴파일, 훈련
        model.fit(x, y)
        #4. 평가, 예측
        results = model.score(x,y)
        print(model_name_list[j], 'model.score :' ,results)
        y_pred = model.predict(x)
        acc = accuracy_score(y,y_pred)
        print(model_name_list[j], 'accuaracy_score :', acc)
#model.score 랑 accuaracy_score가 동일한 값이 나온다.

# iris :
# LinearSVC 0.9666666666666667
# LogisticRegression 0.9733333333333334
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# =============================
# breast_cancer :
# LinearSVC 0.9367311072056239
# LogisticRegression 0.9472759226713533
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# =============================
# digits :
# LinearSVC 0.9782971619365609
# LogisticRegression 1.0
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# =============================
# covtype
# LinearSVC 0.2878511975656269
# LogisticRegression 0.6202797876808052
# DecisionTreeClassifier 1.0
# RandomForestClassifier 0.9999982788651526
# =============================
# load_wine
# LinearSVC 0.9269662921348315
# LogisticRegression 0.9662921348314607
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0