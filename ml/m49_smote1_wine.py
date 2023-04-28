import numpy as np
import pandas as pd
from sklearn.datasets import load_wine # 증폭시킬려면 쪼개서 삭제를 해야하기 때문에
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape) # (178, 13) (178,) y가 벡터형태지
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    48

print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25] # y를 자를려면 x를 잘라야지?
y = y[:-25]
print(x.shape, y.shape) # (153, 13) (153,)
print(y) # 줄어든거 확인 할 수 있지
print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    23

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=377,
    stratify=y # 3/4만큼 각 라벨별로 잘리게끔 하는거야
)
print(pd.Series(y_train).value_counts().sort_index())
# 0    44
# 1    53
# 2    17


#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=377)

#3. 훈련
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score :', score)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ',f1_score(y_test, y_predict, average='macro'))
# print('f1_score(micro) : ',f1_score(y_test, y_predict, average='micro'))
# print('f1_score(micro) : ',f1_score(y_test, y_predict)) # 에러발생

# model.score : 0.9487179487179487
# accuracy_score :  0.9487179487179487
# f1_score(macro) :  0.9439984430496765
# f1_score(micro) :  0.9487179487179487

# f1_score는 암환잔지 아닌지 맞추는 진짜 예측한게 몇명이나 되느냐?
# 이진분류에서 한다 f1_score도 높으면 장땡이야?
# average='micro'// average='macro' 다중분류에서 쓸 수 있게 만들어놓는거야
# 통상 macro 옵션 많이 쓴다 //정밀도 재현률

# 가장 쉬운 증폭은 카피야?

print("==========SMOTE 적용 후============")
smote = SMOTE(random_state=377, k_neighbors=8) # 디폴트 5
# k_neighbors 가까이 있는 거에 생성을 한다는 얘기다
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape) # (159, 13) (159,)
print(pd.Series(y_train).value_counts().sort_index())
# 0    53
# 1    53
# 2    53
# 53보다 더 늘리면 과적합이 일어나지 않을까?

#2-2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=377)

#3-2. 훈련
model.fit(x_train, y_train)

#4-2. 평가
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)
print('model.score :', score)
print('accuracy_score : ', accuracy_score(y_test, y_predict))
print('f1_score(macro) : ',f1_score(y_test, y_predict, average='macro'))