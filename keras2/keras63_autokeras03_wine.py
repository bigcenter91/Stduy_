import autokeras as ak
import numpy as np
from sklearn.datasets import load_wine
from keras.utils.np_utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split


#1. 데이터
x, y = load_wine(return_X_y=True)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=524)

# AutoKeras 분류 모델 생성
model = ak.StructuredDataClassifier(max_trials=1, overwrite=False)  # 최대 시도 횟수 지정

# 모델 훈련
model.fit(x_train, y_train, epochs=100)


best_model = model.export_model()
print(best_model.summary())

# 모델 평가
results = model.evaluate(x_test, y_test)
print('결과:', results)