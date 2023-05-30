from sklearn.datasets import fetch_california_housing, load_iris
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
datasets = load_iris()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target # 타겟 컬럼이 y값이다
# print(df)


y = df['target']
x = df.drop(['target'], axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)