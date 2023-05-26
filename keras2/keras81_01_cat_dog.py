# 최종 지표인 acc 도출!!
# 기존꺼와 전이학습의 성능비교
# 무조건 전이학습 이겨야한다!!!
# 본인사진 넣어서 개인지 고양인지 구별

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions


#1. 데이터

path_cat = 'c:/study_data/_save/cat_dog/'

cd_x_train = np.load(path_cat+ 'keras56_cat_dog_x_train.npy')
cd_x_test = np.load(path_cat + 'keras56_cat_dog_x_test.npy')
cd_y_train = np.load(path_cat + 'keras56_cat_dog_y_train.npy')
cd_y_test = np.load(path_cat + 'keras56_cat_dog_y_test.npy')

print(cd_x_train.shape, cd_y_train.shape) # (3500, 100, 100, 3) (3500,)


cd_y_train = to_categorical(cd_y_train)
cd_y_test = to_categorical(cd_y_test)

#2. 모델 구성
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# vgg19.trainable = False

x = vgg19.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(2, activation='softmax')(x)

model = Model(inputs=vgg19.input, outputs=output1)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(cd_x_train, cd_y_train, epochs=30, batch_size=512, verbose=1)


#4. 평가, 예측
loss = model.evaluate(cd_x_test, cd_y_test)
y_predict = model.predict(cd_x_test)

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(cd_y_test, axis= 1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)


path = 'C:\study_data\_data\\my.png'
img = image.load_img(path, target_size=(100, 100))

x_img = image.img_to_array(img)
print(x_img, '\n', x_img.shape)
print(np.min(x_img), np.max(x_img)) # 0.0 255.0

print("======================= model.predict(x) ====================")
x_img = x_img.reshape(x_img.shape[0], x_img.shape[1], x_img.shape[2])
print(x_img.shape)

x_img = np.expand_dims(x_img, axis=0)
print(x_img.shape)

x_img = preprocess_input(x_img)
print(x_img.shape)
print(np.min(x_img), np.max(x_img))

x_img_pred = model.predict(x_img)
predicted_class = np.argmax(x_img_pred, axis=1)

if predicted_class == 0:
    print("The image contains a cat.")
else:
    print("The image contains a dog.")


print(x_img_pred, '\n', x_img_pred.shape)

print("결과는 :", decode_predictions(x_img_pred, top=5)[0])
