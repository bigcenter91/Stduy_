import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

#1.데이터
x,y = load_breast_cancer(return_X_y=True)

# 필요 없는 특성 삭제
x= np.delete(x, [4, 5, 8, 9, 11, 12], axis=1)

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
print(type(model).__name__, ":", model.feature_importances_)

# 지우기 전
# [0.03416943 0.01029668 0.0531388  0.04794615 0.00604881 0.00474555
#  0.05912765 0.08021216 0.00242332 0.00352914 0.00920663 0.00476516
#  0.01779825 0.05119018 0.00395466 0.00484601 0.00391264 0.00702334
#  0.00358703 0.00480031 0.19453253 0.01563265 0.09068097 0.08151084
#  0.01306031 0.01546865 0.03063698 0.12249818 0.01515726 0.00809974] acc :  0.956140350877193 accuracy_score :  0.956140350877193

# 지운 후
# [0.0373298  0.01562401 0.03201154 0.08109213 0.04130744 0.07274527
#  0.013475   0.03335948 0.00279937 0.00491439 0.01096088 0.00796756
#  0.00364423 0.00358318 0.16131276 0.02022896 0.15629988 0.11585245
#  0.01426723 0.00671164 0.04282752 0.10037334 0.01631012 0.00500182] acc :  0.9649122807017544 accuracy_score :  0.9649122807017544
