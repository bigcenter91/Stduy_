import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from sklearn.datasets import load_wine, fetch_covtype

#1. 데이터



diabetes_path = './_data/dacon_diabetes/'
wine_path = './_data/dacon_wine/'

diabetes_train = pd.read_csv(diabetes_path + 'train.csv', index_col = 0).dropna()
wine_train = pd.read_csv(wine_path + 'train.csv', index_col = 0).dropna()

dia_x = diabetes_train.drop(['Outcome'], axis=1)
dia_y = diabetes_train['Outcome']

wi_x = wine_train.drop(['quality'], axis=1)
wi_y = wine_train['quality']


data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype,
             diabetes_train, wine_train]


lda = [LinearDiscriminantAnalysis()]

model_list = [RandomForestClassifier()]


data_name_list = ['아이리스 : ',
                  '캔서 : ',
                  '디지츠 : ',
                  '와인 : ',
                  '패치코프 : ',
                  '디아벳 : ',
                  '와인 : ']

lda_name = ['엘디에이']

model_name_list = ['랜덤포레스트']

#2. 모델링
model_list = [RandomForestClassifier()]

for model in model_list:
    for i, data in enumerate(data_list):
        if data in [diabetes_train, wine_train]:
            x_train, x_test, y_train, y_test = train_test_split(dia_x, dia_y, test_size=0.2, random_state=42) if data is diabetes_train \
                                              else train_test_split(wi_x, wi_y, test_size=0.2, random_state=42)
        else:
            X, y = data(return_X_y=True)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for j, lda_model in enumerate(lda):
            lda_model.fit(x_train, y_train)
            lda_score = lda_model.score(x_test, y_test)
            print(data_name_list[i] + lda_name[j] + " acc : ", lda_score)

        for k, model in enumerate(model_list):
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            print(data_name_list[i] + model_name_list[k] + " acc : ", score)