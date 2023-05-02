import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#1. 데이터

datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

print(x.shape,y.shape) #(569, 30) (569,)

pca = PCA(n_components= 30) #증폭이 안됨.
x = pca.fit_transform(x)
print(x.shape) #(569, 30)

pca_EVR = pca.explained_variance_ratio_ #EVR임

print(pca_EVR) # pca개수 순서
# [9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
#  8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
#  3.44135279e-07 1.86018721e-07 6.99473205e-08 1.65908880e-08
#  6.99641650e-09 4.78318306e-09 2.93549214e-09 1.41684927e-09
#  8.29577731e-10 5.20405883e-10 4.08463983e-10 3.63313378e-10
#  1.72849737e-10 1.27487508e-10 7.72682973e-11 6.28357718e-11
#  3.57302295e-11 2.76396041e-11 8.14452259e-12 6.30211541e-12
#  4.43666945e-12 1.55344680e-12]

print(sum(pca_EVR)) # 0.9999999999999998

print(np.cumsum(pca_EVR)) #누적합. : pca15개 했을때 원본과 동일한 값.

# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        ]

# 9.82044672e-01 + 1.61764899e-02 = 0.99822116
# 15개 이후로 데이터의 손실이 없다.

pca_cumsum = np.cumsum(pca_EVR)

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()

