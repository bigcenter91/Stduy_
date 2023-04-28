import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, fetch_covtype, load_digits
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error,r2_score
import warnings
warnings.filterwarnings(action='ignore')
import sklearn as sk 
print(sk.__version__) #1.2.2


#1. 데이터 
path = 'c:/stduy_/_data/dacon_wine/'
path_save = 'c:/stduy_/_save/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv) #[5497 rows x 13 columns]
print(train_csv.shape) #(5497,13)
 
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) #[1000 rows x 12 columns] / quality 제외 (1열)

#labelencoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_csv['type'])
aaa = le.transform(train_csv['type'])
print(aaa)   #[1 0 1 ... 1 1 1]
print(type(aaa))  #<class 'numpy.ndarray'>
print(aaa.shape)
print(np.unique(aaa, return_counts=True))

train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis=1)
print(x.shape)                       #(5497, 12)
y = train_csv['quality']

#1-2 one-hot-encoding
print('y의 라벨값 :', np.unique(y))  #[3 4 5 6 7 8 9]
print(np.unique(y, return_counts=True)) # array([  26,  186, 1788, 2416,  924,  152, 5]

import pandas as pd
y=pd.get_dummies(y)
y = np.array(y)
print(y.shape)                       #(5497, 7)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, #stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 10000,
              'learning_rate' : 0.01,
              'max_depth': 3,
              'gamma': 0,
              'min_child_weight': 0,
              'subsample': 0.4,
              'colsample_bytree': 0.8,
              'colsample_bylevel': 0.7,
              'colsample_bynode': 0,
              'reg_alpha': 1,
              'reg_lambda': 1,
              'random_state' : 123,
            #   'eval_metric' : 'error'
              }


#2. 모델
model = XGBClassifier(**parameters)
model.set_params(early_stopping_rounds =100, eval_metric = 'error', **parameters)    


# train the model
model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
          verbose =0
          )

#4. 평가
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
print(f"accuracy_score: {acc}")
print(f"RMSE: {rmse}")
print("======================================")

##################################################
print("컬럼 중요도:",model.feature_importances_)
thresholds = np.sort(model.feature_importances_)        #list형태로 들어가있음. #np.sort오름차순 정렬
print("컬럼 중요도(오름차순):", thresholds)
print("======================================")

from sklearn.feature_selection import SelectFromModel 
for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit=False)    # prefit = False면 다시 훈련 / True :사전훈련 

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print("변형된 x_train:", select_x_train.shape, "변형된 x_test:", select_x_test.shape)

    selection_model = XGBClassifier()

    selection_model.set_params(early_stopping_rounds =10, eval_metric = 'error', **parameters)

    selection_model.fit(select_x_train, y_train,
                        eval_set = [(select_x_train, y_train), (select_x_test, y_test)],
                        verbose =0)
    
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)

    print("Tresh=%.3f, n=%d, acc: %.2f%%" %(i, select_x_train.shape[1], score*100))