import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# Y는 압축할 필요없겠지? // X만 압축을 한다
# target 값은 통상 y라고 할 수 있겠지?
# 차라리 차원축소라고 하면 다른 사람이 이해한다
# target값이 없다 비지도 target값을 생성하지 그럼 비지도


#1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape)
# (569, 30) (569,)

pca = PCA(n_components=30) # 차원을 줄이는 갯수가 된다_n_components
x = pca.fit_transform(x)
print(x.shape) # (569, 30)

pca_EVR = pca.explained_variance_ratio_
# 설명 가능한
print(pca_EVR)
print(sum(pca_EVR)) # 0.9999999999999998 // 뭔가 원핫삘 느껴지니?

pca_cumsum = np.cumsum(pca_EVR) # cumsum : 누적합
print(pca_cumsum)
# [0.98204467 0.99822116 0.99977867 0.9998996  0.99998788 0.99999453
#  0.99999854 0.99999936 0.99999971 0.99999989 0.99999996 0.99999998
#  0.99999999 0.99999999 1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.
#  1.         1.         1.         1.         1.         1.        
# 처음부터 pca를 한개만 했을 때 15개를 했을 때 데이터의 손실이 없을거라고 판단할 수 있다
# 이것도 100% 신뢰하면 안된다

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()