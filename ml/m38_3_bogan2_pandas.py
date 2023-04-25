import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)
#       0    1     2    3
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

#0. 결측치 확인
print(data.isnull()) #True가 결측치야
print(data.isnull().sum())
print(data.info())

#1. 결측치 삭제
print("=============== 결측치 삭제 ================")
# print(data['x1'].dropna()) # 요렇개 하면 그 열에서만 삭제되서 그게 그거다
print(data.dropna()) # 디폴트가 행 위주 삭제다
print("=============== 결측치 삭제 ================")
print(data.dropna(axis=0)) # 행 위주 삭제
print("=============== 결측치 삭제 ================")
print(data.dropna(axis=1)) # 열 위주 삭제
# 무조건 눈으로 확인하는게 가장 좋다

#2-1. 특정값 - 평균
print("=============== 결측치 처리 mean() ================")
means = data.mean() # 각 컬럼별로 때려준다
print('평균 : ', means)
data2 = data.fillna

#2-2. 특정값 - 중위값
print("=============== 결측치 처리 median() ================")
median = data.median()
print('중위값 : ', median)
data3 = data.fillna(median)
print(data3)

#2-3. 특정값 - ffill, bfill
print("=============== 결측치 처리 ffill, bfill() ================")
data4 = data.fillna(method='ffill')
print(data4)
data5 = data.fillna(method='bfill')
print(data5)

#2-4. 특정값 - 임의 값으로 채우기
print("=============== 결측치 처리 - 임의 값으로 채우기 ================")
data6 = data.fillna(777777)
print(data6)

######################### 특정 컬럼만 !!! #########################

#1. x1 컬럼에 평균값을 넣고
data_x1 = data['x1'].mean()
data['x1'] = data['x1'].fillna(data_x1)
print(data)

#2. x2 컬럼에 중위값을 넣고
median_x2 = data['x2'].median()
data['x2'] = data['x2'].fillna(median_x2)
print(data)

#3. x4 컬럼에 ffill한 후 / 제일 위에 남은 행에 777777로 채우기
data['x4'] = data['x4'].fillna(method='ffill')
data['x4'] = data['x4'].fillna(777777)
print(data)
