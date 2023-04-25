# Linear Discriminant Analysis, LDA : 얘도 차원축소
# pca는 데이터의 방향성에 선을 긋는다 평탄화 시킨다음 지표를 다시 잡고_비지도 학습
# lda는 각 데이터의 클래스별로 매칭을 시킨다_지도 학습: y의 클래스를 알기 때문에 또 갈리는게 명확하게 갈린다
# 컬럼의 갯수가 클래스의 갯수보다 작을 때
# 디폴트로 돌아가냐? or 

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.datasets import fetch_covtype, load_digits, load_wine
from tensorflow.keras.datasets import cifar100

#1. 데이터
# x, y = load_iris # 150, 4 > 150, 2
# x, y = load_breast_cancer
# x, y = load_digits
# x, y = load_wine
# x, y = fetch_covtype

data_list = [load_iris,
             load_breast_cancer,
             load_digits,
             load_wine,
             fetch_covtype]

lda = LinearDiscriminantAnalysis()

for i, v in enumerate(data_list):
    x, y = v(return_X_y=True)
    x_lda = lda.fit_transform(x, y)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(data_list[i].__name__,f'cumsum : {cumsum}, \nx.shape : {x.shape}, \nx.lda.shape : {x_lda.shape}')
    
 


# x_lda = lda.fit_transform(x, y)
# print(x_lda.shape) # (50000, 97)
# # LDA가 pca보다 더 좋은 경우가 있어

# lda_EVR = lda.explained_variance_ratio_

# cumsum = np.cumsum(lda_EVR)
# print(cumsum)


