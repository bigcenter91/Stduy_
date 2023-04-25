import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

#1 데이터
data_list = [load_iris(return_X_y = True),
             load_breast_cancer(return_X_y = True),
             load_wine(return_X_y = True),
             load_digits(return_X_y = True)]

data_list_name = ['아이리스',
                  '캔서',
                  '와인',
                  '디지트']

scaler = StandardScaler()



n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 123)


for i, v in enumerate(data_list):
    
    x, y = v
    x = scaler.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    allAlgorithms = all_estimators(type_filter = 'classifier')

    max_score = 0
    max_name = '바보'

    for (name, algorithms) in allAlgorithms:
        try:
            model = algorithms()
            
            scores = cross_val_score(model, x, y, cv = kfold)
            results = round(np.mean(scores), 4)
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            
            if max_score < results:
               max_score = results
               max_score = name
            
        except:
            continue
    
print("==============", data_list_name[i], "===============")
print("최고모델 : ", max_name, max_score)
print("==================================================")

# y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# acc = accuracy_score(y_test, y_predict)
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)