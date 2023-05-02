import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터

datasets = load_iris()

print(datasets.feature_names) #판다스에서는 colunms
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names) #컬럼이름을 넣는 과정
#print(df)

df['target(y)'] = y
print(df) #[150 rows x 5 columns]

print("======================================상관계수 히트 맵 짜잔 =================================================")
print(df.corr())

# ======================================상관계수 히트 맵 짜잔 =================================================
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target(y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658 
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# target(y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

#첫번째 y와의 상관관계를 봐야됨 y와 상관관계가 낮으면 그 컬럼을 제거하는게 나음.

import matplotlib.pyplot as plt
import seaborn as sns #좀더 이쁘게 한다

sns.heatmap(data=df.corr(), 
            square=True, 
            annot=True,
            cbar=True)

plt.show()

#petal width, petal length가 같이 따라다니는 놈이기 때문에 지우면 더 좋아질수도 있음.