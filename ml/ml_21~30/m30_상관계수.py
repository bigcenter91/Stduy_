import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
print(datasets.feature_names) # np: feature_name // pd: columns
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

df = pd.DataFrame(x, columns=datasets.feature_names)
# numpy로 되있지
# print(df)

df['Target(Y)'] = y
print(df) # [150 rows x 5 columns]

print("====================== 상관계수 히트 맵 짜잔 ======================")
print(df.corr())
#단순 회귀
# # y와 상관관계 확인 // sepal length, width는 제거해도 될거 같지?
# petal length, width도 삭제할 고민을 해볼 수 있다
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)
plt.show()