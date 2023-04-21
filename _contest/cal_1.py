import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#1. 데이터
path= "c:/study_data/_data/cal/"

train = pd.read_csv(path + 'train.csv', index_col=0)
test = pd.read_csv(path + 'test.csv', index_col=0)

train = train.fillna(0)

print(train.columns)
print(train)

# Index(['Exercise_Duration', 'Body_Temperature(F)', 'BPM', 'Height(Feet)',
#        'Height(Remainder_Inches)', 'Weight(lb)', 'Weight_Status', 'Gender',
#        'Age', 'Calories_Burned'],
#       dtype='object')

x = train.drop(['Calories_Burned'], axis=1)
y = train['Calories_Burned']

print(x.shape, y.shape) # (7500, 9) (7500,)

# 'Weight_Status'와 'Gender'는 범주형(categorical) 데이터
# encoder = OneHotEncoder(sparse=False)
# x_train_encoded = pd.DataFrame(encoder.fit_transform(x_train[['Gender', 'Weight_Status']]))
# x_train_encoded.columns = encoder.get_feature_names(['Gender', 'Weight_Status'])
# x_train = pd.concat([x_train.drop(['Gender', 'Weight_Status'], axis=1), x_train_encoded], axis=1)

# x_test_encoded = pd.DataFrame(encoder.transform(x_test[['Gender', 'Weight_Status']]))
# x_test_encoded.columns = encoder.get_feature_names(['Gender', 'Weight_Status'])
# x_test = pd.concat([x_test.drop(['Gender', 'Weight_Status'], axis=1), x_test_encoded], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=333, test_size=0.2
)


#2. 모델
model = Sequential()
model.add(Dense(64, input_dim=12, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

#4. 평가, 예측
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('RMSE: ', rmse)
