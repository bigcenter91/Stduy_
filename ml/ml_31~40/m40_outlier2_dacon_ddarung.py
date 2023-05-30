import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import KNNImputer

path = './_data/ddarung/'                                       
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

train_csv = train_csv.fillna(train_csv.median())
test_csv = test_csv.fillna(test_csv.median())   

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    
    print('1사 분위 :', quartile_1)
    print('q2 :', q2)
    print('3사 분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr :', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    return np.where((data_out > upper_bound) | (data_out < lower_bound))[0] # 괄호안의 만족하는 것을 반환한다

train_csv = np.array(train_csv)
                                                                                           

outliers_train = outliers(train_csv)
outliers_test = outliers(test_csv)

print(outliers_train.shape)

train_csv = np.array(train_csv)
test_csv = np.array(test_csv)

# print(train_csv.shape)



# print(train_csv.shape) # (1459, 10)
# print(test_csv.shape) # (715, 9)
# print(train_csv)  
# print(train_csv.isnull().sum())  
# print(test_csv.isnull().sum())  


