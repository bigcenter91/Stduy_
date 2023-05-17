import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

#0. fix seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

#1. 데이터
path = 'C:/study_data/_data/dacon_crime/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

x = train_csv.drop(['TARGET'], axis = 1)
y = train_csv['TARGET']

# 범주형 변수 리스트
qual_col = ['요일', '범죄발생지']

# 원-핫 인코딩
x = pd.get_dummies(x, columns=qual_col)
test_csv = pd.get_dummies(test_csv, columns=qual_col)

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 524
)

model = CatBoostClassifier(
    iterations = 10000,
    depth = 14,  # 기본값: 6
    learning_rate = 0.000001,
    l2_leaf_reg = 0.05,
    one_hot_max_size = 20,
    random_strength = 0.01,
    bagging_temperature = 0.9,
    border_count = 254,    # 기본값: 254
    # verbose=0,
    task_type="GPU",
    )

# model = LGBMClassifier(num_leaves=3000,
#                        max_depth=6,
#                        learning_rate=0.002,
#                        n_estimators=2000,
#                        subsample_for_bin=200000,
#                        min_child_samples= 30,
#                        reg_alpha=0.2,
#                        reg_lambda=0.2,
#                        colsample_bytree=0.9,
#                        subsample=0.8)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
f1 = f1_score(y_test, y_predict, average='macro')
print("f1score : ", f1)

#time
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# Submission
save_path = 'C:/study_data/_save/dacon_crime/'
y_submit= model.predict(test_csv)
sample_submission_csv = pd.read_csv(path + 'sample_submission.csv')
sample_submission_csv[sample_submission_csv.columns[-1]]= y_submit
sample_submission_csv.to_csv(save_path + 'crime_' + date + '.csv', index=False)