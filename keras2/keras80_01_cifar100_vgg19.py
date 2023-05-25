# 10개 파일 뿌라스로 맹그러봐
# 쉐이프 오류인거는 내용 명시하고 추가 모델 맹그러

# 01. VGG19
# 02. Xception
# 03. Resnet50
# 04. InceptionV3
# 05. InceptionResNetV2
# 06. DenseNet121
# 07. MobileNetV2
# 08. NasNetMobile
# 10. EfficeintNetB0

# 공통 Fully커넥티드Layer 구성하지 말고
# GAP로 바로 출력 때릴 것!!!

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score



#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(50000, 32, 32, 3) 
x_test = x_test.reshape(10000, 32, 32, 3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape)

#2. 모델 구성
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

x = vgg19.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(100, activation='softmax')(x)

model = Model(inputs=vgg19.input, outputs=output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(x_train, y_train, epochs=30, batch_size=128, verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)