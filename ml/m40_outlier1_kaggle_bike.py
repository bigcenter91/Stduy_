import numpy as np
from sklearn.covariance import EllipticEnvelope
import pandas as pd
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']
#
imputer = IterativeImputer(estimator=XGBRegressor())
x = imputer.fit_transform(x)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75], axis=0)
    print('1사분위 : ', quartile_1) 
    print('q2 : ', q2) 
    print('3사분위 : ', quartile_3) 
    iqr = quartile_3 - quartile_1 
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
outliers_loc = outliers(x)
print('이상치의 위치 : ', list((outliers_loc)))
x[outliers_loc] = 99999999999
# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()

xgb = XGBRegressor()
xgb.fit(x, y)
results = xgb.score(x,y)
y_submit = xgb.predict(test_csv)
print(results)

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + 'dacon_kaggle_submit.csv')