import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1.데이터
x,y = load_digits(return_X_y=True)

# 필요 없는 특성 삭제
x= np.delete(x, [0,32,37,40], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model = RandomForestClassifier()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)
print("=====================================================")
#print(type(model).__name__, ":", model.feature_importances_)
import pandas as pd
#argmin = np.argmin(model.feature_importances_, axis = 0, k=4)
argmin = np.argpartition(model.feature_importances_, 4)[:4]

x_drop = pd.DataFrame(x).drop(argmin, axis = 1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 123)

model.fit(x_train1, y_train1)

result = model.score(x_test1, y_test1)
print( 'result : ', result)

y_predict1 = model.predict(x_test1)

acc1 = accuracy_score(y_test1, y_predict1)
print( 'acc1 : ', acc1)

# import heapq
# small_nums = heapq.nsmallest(5, model.feature_importances_)
# print(small_nums)

# 변경전
# accuracy_score :  0.9722222222222222
# [0.00000000e+00 2.49896106e-03 2.02285052e-02 9.64088217e-03
#  1.00099671e-02 1.93174429e-02 8.88437018e-03 4.11691734e-04
#  1.15909704e-04 9.78317339e-03 2.62275812e-02 7.74516467e-03
#  1.32913347e-02 2.87640903e-02 5.14974307e-03 5.87909128e-04
#  1.23806767e-04 7.86808687e-03 1.92380382e-02 2.92320104e-02
#  3.07689110e-02 4.79115123e-02 1.03301530e-02 3.83338226e-04
#  9.73053560e-05 1.28317751e-02 4.41676466e-02 2.26585627e-02
#  2.57580856e-02 2.67127592e-02 2.62402326e-02 1.14111521e-05
#  0.00000000e+00 3.58882469e-02 2.51786372e-02 1.64612948e-02
#  4.40156363e-02 1.89359198e-02 2.32200861e-02 0.00000000e+00
#  5.31011074e-05 8.79321428e-03 4.18124995e-02 4.03558901e-02
#  2.19786455e-02 1.90066161e-02 1.89549791e-02 7.40397431e-05
#  3.03049192e-05 2.27965933e-03 1.55741933e-02 2.18750843e-02
#  1.38676045e-02 2.57016197e-02 2.28228534e-02 1.78536351e-03
#  2.71756259e-05 1.61060303e-03 2.17671304e-02 1.15510100e-02
#  2.76714051e-02 3.13434200e-02 1.71002434e-02 3.27316151e-03]
# 
# [0.0, 0.0, 0.0, 1.1411152149965884e-05, 2.7175625861164567e-05]

# [1.75817667e-03 1.85268003e-02 1.10575111e-02 9.72017834e-03
#  2.08117433e-02 9.40929875e-03 8.79757839e-04 7.57547270e-05
#  9.03560752e-03 2.50616130e-02 7.45218582e-03 1.66600583e-02
#  2.74946881e-02 4.79997722e-03 4.79733873e-04 7.94159518e-05
#  9.39041263e-03 2.17937481e-02 2.48421379e-02 2.81926716e-02
#  4.81246788e-02 8.28664749e-03 2.24536950e-04 5.60970462e-05
#  1.59610865e-02 4.28846039e-02 2.51340585e-02 3.54356476e-02
#  2.46533718e-02 2.84012944e-02 2.06359891e-06 3.15195807e-02
#  3.16602380e-02 1.90086225e-02 3.70782199e-02 2.70737411e-02
#  0.00000000e+00 1.11767287e-02 3.48694793e-02 4.51537326e-02
#  2.13859761e-02 2.12716195e-02 1.90447298e-02 1.93036051e-04
#  0.00000000e+00 2.71795683e-03 1.83156534e-02 2.25855593e-02
#  1.30409263e-02 2.43785317e-02 2.93502965e-02 2.05348147e-03
#  1.33523341e-05 1.85562247e-03 2.21117366e-02 1.14655513e-02
#  2.57395155e-02 3.11241521e-02 1.55177645e-02 3.60863778e-03] 0.9722222222222222