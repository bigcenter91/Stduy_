import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


#1. 데이터
x, y = load_digits(return_X_y=True)

pf = PolynomialFeatures(degree=2)
x_pf = pf.fit_transform(x)
print(x_pf.shape) # (150, 15)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, random_state=123, train_size=0.8, shuffle=True,
    stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))

# score :  0.9888888888888889
# acc :  0.9888888888888889