import pandas as pd
df = pd.DataFrame({'A' : [1, 2, 3, 4, 5],
                   'B' : [10, 20, 30, 40, 50],
                   'C' : [5, 4, 3, 2, 1]})
# 딕셔너리 형태가 된거지
print(df)

correlations = df.corr()
print(correlations)
# 너무 신뢰하면 안되 하지만 참고는 할 수 있겠지?