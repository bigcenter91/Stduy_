import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Input, Dropout
from tensorflow.keras.optimizers import Adam
# node와 lr 적용해서 소스 완성
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(x_test.shape[0], -1).astype('float32')/255.    #(10000,784)

#2. 모델
def build_model(drop=0.5, optimizer='adam', activation='relu',
                node1=64, node2=64, node3=64, node4=128, lr=0.001):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(node1, activation=activation, name='hidden1')(inputs)
    x = Dense(drop)(x)
    x = Dense(node2, activation=activation, name='hidden2')(x)
    x = Dense(drop)(x)
    x = Dense(node3, activation=activation, name='hidden3')(x)
    x = Dense(drop)(x)
    x = Dense(node4, activation=activation, name='hidden4')(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    
    model = Model(inputs = inputs, outputs = outputs)
    
    model.compile(optimizer = optimizer, metrics=['acc'],
                  loss = 'sparse_categorical_crossentropy')
    return model

def create_hyperparamter():
    batchs = [100, 200, 300, 400, 500]
    optimizers = ['adam', 'rmsprop', 'adadelta', 'adagrad']
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    nodes = [128, 32, 64, 16]
    dropout = [0.2, 0.3, 0.4, 0.5]
    activation = ['relu', 'elu', 'selu', 'linear']
    return {'node1': nodes,
            'node2': nodes,
            'node3': nodes,
            'node4': nodes,
            'batch_size' : batchs,
            'optimizer' : optimizers,
            'lr' : learning_rates,
            'drop' : dropout,
            'activation' : activation,
            }
    
hyperparameters = create_hyperparamter()
print(hyperparameters)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
keras_model = KerasClassifier(build_fn=build_model, verbose=1) #epochs=3)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# model1 = build_model()
# model = GridSearchCV(keras_model, hyperparameters, cv=3)
model = RandomizedSearchCV(keras_model, hyperparameters, cv=3, n_iter=2, verbose=1)

callbacks = [
    EarlyStopping(patience=3, monitor="val_loss"),
    ModelCheckpoint(filepath='./_save/MCP/keras66_ModelCheckPoint.hdf5', save_best_only=True, monitor="val_loss", mode="min", verbose=1),
]


import time
start = time.time()
model.fit(x_train, y_train, epochs=10, callbacks = callbacks, validation_split=0.2)
end = time.time()

print("걸린시간 :", end-start)
print("model.best_params_ :", model.best_params_)
print("model.best_estimator_ :", model.best_estimator_)
# params와 estimator 차이 확인해봐
print("model.best_score_ :", model.best_score_)             # train에 대한 스코어
print("model.score :", model.score(x_test, y_test))         # test에 대한 스코어

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print('acc_score :', accuracy_score(y_test, y_predict))


# 걸린시간 : 6.334747314453125
# model.best_params_ : {'optimizer': 'adam', 'drop': 0.2, 'batch_size': 500, 'activation': 'linear'}
# model.best_estimator_ : <keras.wrappers.scikit_learn.KerasClassifier object at 0x000001EAD335CE80>
# model.best_score_ : 0.11236666887998581
# 20/20 [==============================] - 0s 2ms/step - loss: 2.3014 - acc: 0.1135
# model.score : 0.11349999904632568
# acc_score : 0.1135