import time
import autokeras as ak
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype, fetch_california_housing, load_diabetes]

# 1. 데이터
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

    # 2. 모델
    if i < 5:
        model = ak.StructuredDataClassifier(overwrite=False,max_trials=2)
    else:
        model = ak.StructuredDataRegressor(overwrite=False,max_trials=2)

    # 3. 컴파일, 훈련
    model.fit(x_train, y_train, epochs=1, validation_split=0.15)

    # 4. 평가, 예측
    results = model.evaluate(x_test, y_test)
    print(data_list[i].__name__, 'results : ', results)
    best_model = model.export_model()
    
    # path = './_save/autokeras/'
    # best_model.save(path + f'{data_list[i].__name__}_keras63_autokeras.h5')