import numpy as np
import pandas as pd
from datetime import datetime

dates = ['4/25/2023', '4/26/2023', '4/27/2023', '4/28/2023', '4/29/2023', '4/30/2023']

dates = pd.to_datetime(dates)
print(dates)
print(type(dates)) # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

print("===============================")

ts = pd.Series([2, np.nan, np.nan, 8, 10, np.nan], index=dates)
# 벡터와 매칭할 수 있겠지, 1차원이다 모이면 컬럼 하나와 매칭
# pandas는 시리즈와 데이터 프레임이 있다 시리즈가 모이면 데이터 프레임
print(ts)

print("===============================")
# 판다스에는 인덱스와 헤더가 있지
ts = ts.interpolate()
print(ts)