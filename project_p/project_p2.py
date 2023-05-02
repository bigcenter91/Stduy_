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
          
          
          # 1. 데이터

path = "d:/study_data/_data/project_p/"
save_path = "d:/study_data/_save/project_p/"

weather_g_21 = pd.read_csv(path + 'OBS_212208_Gwangju weather_2021.csv', index_col=0, encoding='cp949')
weather_g_22 = pd.read_csv(path + 'OBS_212208_Gwangju weather_2022.csv', index_col=0, encoding='cp949')
weather_j_21 = pd.read_csv(path + 'OBS_212208_Jeonju weather_2021.csv', index_col=0, encoding='cp949')
weather_j_22 = pd.read_csv(path + 'OBS_212208_Jeonju weather_2022.csv', index_col=0, encoding='cp949')
weather_m_21 = pd.read_csv(path + 'OBS_212208_Mokpo weather_2021.csv', index_col=0, encoding='cp949')
weather_m_22 = pd.read_csv(path + 'OBS_212208_Mokpo weather_2022.csv', index_col=0, encoding='cp949')
rice_k_21 = pd.read_csv(path + '2122_Rice production_polished rice_2021.csv', index_col=0, encoding='cp949')
rice_k_22 = pd.read_csv(path + '2122_Rice production_polished rice_2022.csv', index_col=0, encoding='cp949')

# 2. 데이터 전처리

# 각 지역별로 날씨 데이터를 결합합니다.
weather_g = pd.concat([weather_g_21, weather_g_22], axis=0)
weather_j = pd.concat([weather_j_21, weather_j_22], axis=0)
weather_m = pd.concat([weather_m_21, weather_m_22], axis=0)

# 날짜 데이터를 인덱스로 변경합니다.
weather_g.index = pd.to_datetime(weather_g.index)
weather_j.index = pd.to_datetime(weather_j.index)
weather_m.index = pd.to_datetime(weather_m.index)

# 각 지역별로 날씨 데이터를 8월로 필터링합니다.
weather_g = weather_g[weather_g.index.month == 8]
weather_j = weather_j[weather_j.index.month == 8]
weather_m = weather_m[weather_m.index.month == 8]

# 각 지역별로 불필요한 컬럼을 제거합니다.
weather_g.drop(['최저기온(°C)', '최고기온(°C)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)', '일강수량(mm)',
                 '최대 순간 풍속(m/s)', '최대 풍속(m/s)', '평균 풍속(m/s)', '평균 이슬점온도(°C)'], axis=1, inplace=True)
weather_j.drop(['최저기온(°C)', '최고기온(°C)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)', '일강수량(mm)',
                 '최대 순간 풍속(m