import numpy as np
import pandas as pd
import time

from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_diabetes, fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm

# 1 데이터
dacon_diabetes_path = './_data/dacon_diabetes/'
ddarung_path = './_data/ddarung/'
kaggle_bike_path = './_data/kaggle_bike/'

dacon_diabetes = pd.read_csv(dacon_diabetes_path + 'train.csv', index_col=0).dropna()
ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col=0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col=0).dropna()

def load_dacon_diabetes():
    x = dacon_diabetes.drop(['Outcome'], axis=1)
    y = dacon_diabetes['Outcome']
    return x, y

def load_dacon_ddarung():
    x = ddarung.drop(['count'], axis=1)
    y = ddarung['count']
    return x, y

def load_kaggle_bike():
    x = kaggle_bike.drop(['count', 'casual', 'registered'], axis=1)
    y = kaggle_bike['count']
    return x, y

# 위의 코드는 x와 y로 특성을 나누기 위해 삭제해야 하는 특성을 함수로 정의하여 아래의 반복문과 조건문에서 사용하기 쉽도록 도와줍니다.

data_list = {'iris': load_iris,
             'cancer': load_breast_cancer,
             'wine': load_wine,
             'digits': load_digits,
             'diabetes': load_diabetes,
             'california': fetch_california_housing,
             'dacon_diabetes': load_dacon_diabetes,
             'ddarung': load_dacon_ddarung,
             'kaggle_bike': load_kaggle_bike}

scaler_list = {'MinMax': MinMaxScaler,
               'MaxAbs': MaxAbsScaler,
               'Standard': StandardScaler,
               'Robust': RobustScaler}

total_datasets = len(data_list)
total_scalers = len(scaler_list)

for i, d in enumerate(data_list, start=1): # tqdm은 현재 코드의 진행 상태를 터미널 창에 퍼센트로 표시해줍니다.
    print(f'데이터 진행 상태: {d} ({i/total_datasets * 100:.1f}%)') # 데이터의 반복 상태를 백분율로 출력합니다.
    if d in ['iris', 'cancer', 'wine', 'digits', 'diabetes', 'california']:
        x, y = data_list[d](return_X_y=True)
    elif d in ['dacon_diabetes', 'ddarung', 'kaggle_bike']:
        x, y = data_list[d]()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1234)
    
    for j, s in enumerate(scaler_list, start=1): # tqdm은 현재 코드의 진행 상태를 터미널 창에 퍼센트로 표시해줍니다.
        print(f'스케일러 진행 상태: {d} ({j/total_scalers * 100:.1f}%)') # 스케일러의 반복 상태를 백분율로 출력합니다.
        scaler = scaler_list[s]()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    
        # 2 모델 구성
        def build_model(drop=0.5, optimizer='adam', activation='relu', node1=64, node2=64, node3=64, node4=64, lr=0.001):
            inputs = Input(shape=(x_train.shape[1],), name='inputs')
            x = Dense(node1, activation=activation, name='hidden1')(inputs)
            x = Dropout(drop)(x)
            x = Dense(node2, activation=activation, name='hidden2')(x)
            x = Dropout(drop)(x)
            x = Dense(node3, activation=activation, name='hidden3')(x)
            x = Dropout(drop)(x)
            x = Dense(node4, activation=activation, name='hidden4')(x)
            outputs = Dense(1, activation='sigmoid', name='outputs')(x)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])

            return model
        
        def create_hyperparameter():
            batchs = [100, 200, 300, 400, 500]
            optimizers = ['adam', 'rmsprop', 'adadelta']
            drops = [0.2, 0.3, 0.4, 0.5]
            activations = ['relu', 'elu', 'selu']
            nodes = [32, 64, 128, 256]
            learning_rates = [0.001, 0.01, 0.1]
            
            return {'batch_size': batchs, 'optimizer': optimizers, 'drop': drops, 'activation': activations, 'node1': nodes, 'node2': nodes, 'node3': nodes, 'node4': nodes, 'lr': learning_rates}
        
        hyperparameters = create_hyperparameter()
        print(hyperparameters) # {'batch_size': [100, 200, 300, 400, 500], 'optimizer': ['adam', 'rmsprop', 'adadelta'], 'drop': [0.2, 0.3, 0.4, 0.5], 'activation': ['relu', 'elu', 'selu']}

        if d in ['iris', 'cancer', 'wine', 'digits', 'dacon_diabetes', 'ddarung']:
            # Classification task
            keras_model = KerasClassifier(build_fn=build_model, verbose=1)
        else:
            # Regression task
            keras_model = KerasRegressor(build_fn=build_model, verbose=1)

        model = RandomizedSearchCV(keras_model, hyperparameters, cv=2)

        # 3 훈련
        es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
        lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)
        s_time = time.time()
        model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=1, validation_split=0.2, callbacks=[es, lr])
        e_time = time.time()

        # 4 평가, 예측
        if d in ['iris', 'cancer', 'wine', 'digits', 'dacon_diabetes', 'ddarung']:
            # Classification task
            score = model.score(x_test, y_test)
            print('최적의 매개변수: ', model.best_estimator_)
            print('Accuracy: ', score)
        else:
            # Regression task
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            print('최적의 매개변수: ', model.best_estimator_)
            print('R2 Score: ', r2)

        print('걸린 시간: ', e_time - s_time)
        print()

print('======= 모델링 끝 =======')


from tensorflow.python.client import device_lib
device_lib.list_local_devices()