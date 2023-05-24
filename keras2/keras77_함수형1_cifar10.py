# 맹그러!!
# 가중치 동결과 동결하지 않았을 때 성능비교

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

#1. 데이터
(x_train,y_train),(x_test,y_test) = cifar10.load_data()

#2. 모델
input = Input(shape=(32,32,3),name='input')
x = VGG16(weights='imagenet', include_top=False)(input)
x = Flatten()(x)
x = Dense(100,name='fc1')(x)
x = Dense(100,name='fc2')(x)
output = Dense(10,activation='softmax',name='output')(x)

model = Model(inputs=input, outputs=output)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1)


#4. 평가, 예측
model.evaluate(x_test,y_test)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=1)

acc = accuracy_score(y_test, y_predict)

print('acc : ', acc)