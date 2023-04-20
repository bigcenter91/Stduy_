import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.neighbors import LocalOutlierFactor

# 훈련 데이터 및 테스트 데이터 로드
path= "d:/study_data/_data/fac/"
save_path=  "d:/study_data/_save/fac/"
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Preprocess data
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])

print(train_data.columns)
# Index(['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current',
#        'motor_rpm', 'motor_temp', 'motor_vibe', 'type'],
#       dtype='object')

# Select subset of features for LOF model
features = ['motor_current','motor_temp', 'out_pressure', 'motor_rpm']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size= 0.8, random_state= 2222)

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.1)
y_pred_train = lof.fit_predict(X_train)

# Tuning: Adjust the n_neighbors and contamination parameters
lof_tuned = LocalOutlierFactor(n_neighbors=25, contamination=0.048)
y_pred_train_tuned = lof_tuned.fit_predict(X_train)

# Predict anomalies in test data using tuned LOF
test_data_lof = scaler.transform(test_data[features])
y_pred_test_lof = lof_tuned.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

import matplotlib.pyplot as plt
import seaborn as sns

print(test_data.corr())
plt.figure(figsize=(10,8))
sns.set(font_scale=1.2)
sns.heatmap(train_data.corr(), square=True, annot=True, cbar=True)
plt.show()

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)

