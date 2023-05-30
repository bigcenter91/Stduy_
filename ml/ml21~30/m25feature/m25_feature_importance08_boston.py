import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#1.데이터
x,y = load_boston(return_X_y=True)

# 필요 없는 특성 삭제
#x= np.delete(x, 12, axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, shuffle= True, random_state=27
)

#2. 모델
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = r2_score(y_test,y_predict)
print("r2_score : ", acc)
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

acc1 = r2_score(y_test1, y_predict1)
print( 'acc1 : ', acc1)

# acc :  0.9086674963948556
# r2_score :  0.9086674963948556
# =====================================================
# result :  0.8232928936554376
# acc1 :  0.8232928936554376