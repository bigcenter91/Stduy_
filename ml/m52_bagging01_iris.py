# 알지???
# 각각 만들어서 비교

import numpy as np
import numpy as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8, shuffle=True, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
#2. 모델
model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=123,
                          bootstrap=True,
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))

# score :  0.9333333333333333 // DecisionTreeClassifier
# acc :  0.9333333333333333

