# Linear Discriminant Analysis, LDA : 얘도 차원축소
# pca는 데이터의 방향성에 선을 긋는다 평탄화 시킨다음 지표를 다시 잡고_비지도 학습
# lda는 각 데이터의 클래스별로 매칭을 시킨다_지도 학습: y의 클래스를 알기 때문에 또 갈리는게 명확하게 갈린다

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from sklearn.datasets import load_wine, fetch_covtype

#1. 데이터

diabetes_path = './_data/dacon_diabetes/'
wine_path = './_data/dacon_wine/'

diabetes_train = pd.read_csv(diabetes_path + 'train.csv', index_col = 0).dropna()
wine_train = pd.read_csv(wine_path + 'train.csv', index_col = 0).dropna()


data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype,
             diabetes_train, wine_train]


model_list = [RandomForestClassifier()]


data_name_list = ['아이리스 : ',
                  '캔서 : ',
                  '와인 : ',
                  '패치코프 : ',
                  '디아벳 : ',
                  '와인 : ']

model_name_list = ['랜덤포레스트']


# pca = PCA(n_components=3)
# x = pca.fit_transform(x)
# pca 디폴트 전체가 그대로 나오는거다
# print(x.shape) # (150, 3) 원래 150, 4였지?

# lda = LinearDiscriminantAnalysis()
lda = LinearDiscriminantAnalysis(n_components=3)
# n_components는 클래스의 갯수 빼기 하나 이하로 가능하다!!

x_lda = lda.fit_transform(x, y)
print(x_lda.shape) # (150, 2)
# LDA가 pca보다 더 좋은 경우가 있어