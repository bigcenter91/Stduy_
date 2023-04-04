import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
save_path = 'd:/study_data/_save/cifar10/'

x_train = np.load(save_path + 'keras58_cifar10_x_train.npy')
x_test = np.load(save_path + 'keras58_cifar10_x_test.npy')
y_train = np.load(save_path + 'keras58_cifar10_y_train.npy')
y_test = np.load(save_path + 'keras58_cifar10_y_test.npy')

#2. 모델
model = Sequential()
model.add(Conv2D(filters = 20, kernel_size = (2, 2), input_shape = (32, 32, 3)))
model.add(Conv2D(filters = 40, kernel_size = (6, 6)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(40))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['acc'])
es = EarlyStopping(monitor = 'val_acc',
                   patience = 20,
                   mode = 'max',
                   verbose = 1,
                   restore_best_weights = True)
model.fit(x_train, y_train,
          epochs = 1,
          batch_size = 128,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = np.round(model.predict(x_test))

y_test_acc = np.argmax(y_test, axis = 1)
y_predict = np.argmax(y_predict, axis = 1)

acc = accuracy_score(y_test_acc, y_predict)
print('acc : ', acc)