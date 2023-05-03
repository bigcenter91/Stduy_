import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1 데이터
ddarung_path = 'c:/_study/_data/_ddarung/'
kaggle_bike_path = 'c:/_study/_data/_kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1).values
y1 = ddarung['count'].values

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1).values
y2 = kaggle_bike['count'].values

data_list = [load_iris,
             load_breast_cancer,
             load_digits,
             load_wine,
             load_diabetes,
             fetch_california_housing,
             (x1, y1),
             (x2, y2)]

model_list = [RandomForestClassifier(),
              RandomForestRegressor()]

lda = LinearDiscriminantAnalysis()
pca = PCA()

data_name = ['아이리스',
             '캔서',
             '디지트',
             '와인',
             '디아뱃',
             '캘리포니아',
             '따릉이',
             '캐글 바이크']

for i in range(len(data_list)):
    if i < 6:
        x, y = data_list[i](return_X_y = True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    else:
        x, y = data_list[i]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    if i < 4:
        model = model_list[0]
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        x_lda = lda.fit_transform(x, y)
        lda_EVR = lda.explained_variance_ratio_
        cumsum = np.cumsum(lda_EVR)
        print(f'데이터 이름 : {data_name[i]} = 데이터1 {x.shape} -> 데이터2{x_lda.shape}')
        print('acc : ', acc)
        
    elif i == 5 or 6 or 7:
        model = model_list[1]
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        x_pca = pca.fit_transform(x)
        pca_EVR = pca.explained_variance_ratio_
        cumsum = np.cumsum(pca_EVR)
        print(f'데이터 이름 : {data_name[i]} = 데이터1 {x.shape} -> 데이터2{x_pca.shape}')
        print('r2 : ', r2)