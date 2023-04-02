import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


#1. 데이터

path = 'd:/study_data/_save/breed/'

dog_x_train = np.load(path + 'keras56_5_dog_x_train.npy')
dog_x_test = np.load(path + 'keras56_5_dog_x_test.npy')
dog_y_train = np.load(path + 'keras56_5_dog_y_train.npy')
dog_y_test = np.load(path + 'keras56_5_dog_y_test.npy')

print(dog_x_train)
print(dog_x_test.shape, dog_y_test.shape) 
# (300, 200, 200, 3) (300, 5)



model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(500, 500, 4), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Conv2D(16, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(5, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(dog_x_train, dog_y_train, epochs=50,
                 validation_data=(dog_x_test, dog_y_test),
                 validation_steps=0.3,
                 )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# 4. 평가, 예측
loss = model.evaluate(dog_x_test, dog_y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(dog_x_test))
from sklearn.metrics import accuracy_score
acc = accuracy_score(dog_y_test, y_predict)
print('acc : ', acc)

