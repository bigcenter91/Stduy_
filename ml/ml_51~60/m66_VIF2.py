# 다중공선성?

from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler # 통상 스탠다드 스케일러 쓴다 = 다중공선성에서

# 로그변환은 회귀 데이터

#1. 데이터
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target # 타겟 컬럼이 y값이다
# print(df)


y = df['target']
x = df.drop(['target'], axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 다중공선성
vif = pd.DataFrame()
vif['variables'] = x.columns
# vif['vif'] = [variance_inflation_factor]

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
print(vif)
# 다중공선성 확인할 때는 스케일링 해준다
# 처음에 스케일링한다 다음 y 넣지않는다

#     variables       VIF
# 0      MedInc  2.501295
# 1    HouseAge  1.241254
# 2    AveRooms  8.342786
# 3   AveBedrms  6.994995
# 4  Population  1.138125
# 5    AveOccup  1.008324
# 6    Latitude  9.297624
# 7   Longitude  8.962263

x = x.drop(['Latitude', 'Longitude'], axis=1)
print(x)

# 학자마다 약간씩 틀려 5가 좋다 10이 좋다 // 보통 학자들은 5를 잡고 개발자들은 10을 잡는다
# Latitude 하나 날려서 해보고 Longitude 날려서 해보고 아니면 pca로 둘다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, shuffle=True, test_size=0.2,
    # stratify=y
)

scaler2 = StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

#2. 모델
model = RandomForestRegressor(random_state=337)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", results)

# 결과 :  0.8140377627468227
# 결과 :  0.7430084497100284 // Latitude
# 결과 :  0.6889105651586513 // Longitude, Latitude