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
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100

#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
# x, y = load_digits(return_X_y=True)
# digits가 mnist 축소형이지?
# 선 그어서 클래스의 위치만큼 뺏지
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)

x_train = x_train.reshape(50000, 32*32*3)

pca = PCA(n_components=97)
x_train=pca.fit_transform(x_train)
# pca = PCA(n_components=3)
# x = pca.fit_transform(x)

lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=3)
# 통상 n_components는 클래스의 갯수 빼기 하나 이하로 가능하다!!

x_lda = lda.fit_transform(x_train, y_train)
print(x_lda.shape) # (50000, 97)
# LDA가 pca보다 더 좋은 경우가 있어

#실질적으로 성능이 좋다는 얘기는 못한다 2번 엮은거



# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris

# iris = load_iris

# x = iris.data[:, 2:]
# y = iris.target

# plt.scatter(x[:,0], x[:,1], c=y)
# plt.xlabel('petal length')
# plt.ylabel()