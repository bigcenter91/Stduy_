import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df1 = pd.read_csv("지역1_기상데이터_2021_2022_8월.csv")
df2 = pd.read_csv("지역2_기상데이터_2021_2022_8월.csv")
df3 = pd.read_csv("지역3_기상데이터_2021_2022_8월.csv")
df_rice = pd.read_csv("전국_쌀생산량_2021_2022.csv")

# 필요한 feature 선택
df1 = df1[['날짜', '기온', '강수량']]
df2 = df2[['날짜', '기온', '강수량']]
df3 = df3[['날짜', '기온', '강수량']]
df_rice = df_rice[['날짜', '생산량']]

# 컬럼 이름 변경
df1.columns = ['date', 'temp1', 'rain1']
df2.columns = ['date', 'temp2', 'rain2']
df3.columns = ['date', 'temp3', 'rain3']
df_rice.columns = ['date', 'rice']

# 날짜를 datetime 형식으로 변경
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])
df3['date'] = pd.to_datetime(df3['date'])
df_rice['date'] = pd.to_datetime(df_rice['date'])

# 인덱스를 날짜로 변경
df1 = df1.set_index('date')
df2 = df2.set_index('date')
df3 = df3.set_index('date')
df_rice = df_rice.set_index('date')

# 2021년 8월 ~ 2022년 8월 데이터 추출
df1 = df1.loc['2021-08-01':'2022-08-31']
df2 = df2.loc['2021-08-01':'2022-08-31']
df3 = df3.loc['2021-08-01':'2022-08-31']
df_rice = df_rice.loc['2021-08-01':'2022-08-31']

# 결측치 확인
print(df1.isnull().sum())
print(df2.isnull().sum())
print(df3.isnull().sum())
print(df_rice.isnull().sum())

# 데이터 병합
df = pd.concat([df1, df2, df3, df_rice], axis=1)
df = df.dropna()

# 독립 변수와 종속 변수 분리
; X = df[['temp1', 'temp2