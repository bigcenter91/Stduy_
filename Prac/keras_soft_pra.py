from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.utils import to_categorical

# 1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
y = to_categorical(y, num_classes=7)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True,
    test_size=0.2,
    stratify=y,
)

# 2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=54))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(8, activation='softmax'))
model.summary()

# 3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=1,
          validation_split=0.2,
          verbose=1)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
print(results)
print('loss : ', results[0])
print('acc : ', results[1])

y_pred = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred, axis=1)
acc = accuracy_score(y_test_acc, y_pred)
print('accuracy : ', acc)