# Linear Discriminant Analysis, LDA : 얘도 차원축소
# 상민이가 회귀에서 된다고 했어!!!
# 성호는 y에 round 때렸어!!

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_california_housing
from tensorflow.keras.datasets import cifar100

#1. 데이터
# x, y = load_iris(return_X_y=True)
# x, y = load_breast_cancer(return_X_y=True)
# x, y = load_digits(return_X_y=True)
x, y = load_diabetes(return_X_y=True)
# x, y = fetch_california_housing(return_X_y=True)
# y = np.round(y)
print(y) # diabetes은 수치형으로 나오지 소수점 없이?
print(np.unique(y, return_counts=True))
# diabetes은 회귀임에도 정수형이여서 분류 클래스로 본거야 한마디로 잘못 인식한거다
# california에 round 씌워주면 정수형으로 판단해서 나오는거다 = 데이터 조작이 될 수 있다
# 결론은 round를 씌워주면 안되고 회귀는 쓰면 안된다

lda = LinearDiscriminantAnalysis()
# lda = LinearDiscriminantAnalysis(n_components=3)
# 통상 n_components는 클래스의 갯수 빼기 하나 이하로 가능하다!!

x_lda = lda.fit_transform(x, y)
print(x_lda.shape) # (20640, 5)
# LDA가 pca보다 더 좋은 경우가 있어

### 회귀는 원래 안되. 하지만 diabetes는
# 정수형이라서 LDA에서 y의 클래스로 잘못 인식한거야 그래서 돈거야
# 성호는 캘리포니아에서 라운드 처리를 했어
# 그러다보니 그것도 정수형이라서 클래스로 인식되서 돈거야

# 회귀 데이터는 원칙적으로 에러지만
# 위처럼 돌리고 싶으면 돌려도 무관하다
# 성능은 보장 못한다