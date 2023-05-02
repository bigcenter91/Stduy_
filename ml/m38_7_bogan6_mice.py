# MICE(Multiple Imputation by Chained Equations)

import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__)
from impyute.imputation.cs import mice


data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
                    ).transpose()

# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']
print(data)

#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# impute_df = mice(data) # mice에서는 numpy로 넣어줘 그렇지 않으면 밑에 에러발생
# AttributeError: 'DataFrame' object has no attribute 'as_matrix'

# impute_df = mice(data.values)
impute_df = mice(data.to_numpy())
print(impute_df)

# interpolate처럼 선형구조로 한다 두가지 섞어가면서 쓰면 된다
# pandas > numpy = to_numpy
