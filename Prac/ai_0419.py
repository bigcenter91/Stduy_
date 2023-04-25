import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.ensemble import IsolationForest

# 훈련 데이터 및 테스트 데이터 로드
path= "d:/study_data/_data/fac/"
save_path=  "d:/study_data/_save/fac/"
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# Feature Engineering: RPM 특성 생성
train_data['RPM'] = train_data['motor_current'] / train_data['motor_temp']
test_data['RPM'] = test_data['motor_current'] / test_data['motor_temp']

# Select subset of features for Isolation Forest model
features = ['motor_current', 'motor_temp', 'RPM']

# Prepare train and test data
X = train_data[features]

# Split data into train and validation sets
X_train, X_val = train_test_split(X, train_size= 0.8, random_state= 3333)

# Normalize data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Apply Isolation Forest
isof = IsolationForest(n_estimators=25, contamination=0.1, random_state=3333)
isof.fit(X_train)

# Predict anomalies in test data using Isolation Forest
test_data_isof = scaler.transform(test_data[features])
isof_predictions = isof.predict(test_data_isof)
isof_predictions = [1 if x == -1 else 0 for x in isof_predictions]

submission['label'] = pd.DataFrame({'Prediction': isof_predictions})
print(submission.value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + 'submission.csv', index=False)
