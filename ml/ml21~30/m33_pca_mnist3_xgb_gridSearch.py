import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
#n_conponent > 0.95 이상

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
x = x.reshape(x.shape[0], -1)
n_c_list = [154, 331, 486, 713]
pca_list = [0.95, 0.99, 0.999, 1.0]


parameters = [
    {"n_estimators": [100,200,300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
    "max_depth":[4,5,6]},
    {"n_estimators": [90,100,110], "learning_rate" : [0.1, 0.001, 0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators": [90,110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6,0.7,0.9]}
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

for i in range(len(n_c_list)):
    pca = PCA(n_components=n_c_list[i])
    x_p = pca.fit_transform(x.astype('float32'))
    x_train, x_test, y_train, y_test = train_test_split(x_p, y, train_size=0.8, shuffle=True, random_state=123)

    model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0),
                        parameters, cv=kfold, refit=True, n_jobs=-1,verbose=1)
    model.fit(x_train, y_train)
    
    acc = model.score(x_test, y_test)
    print(f'PCA {pca_list[i]} test acc : {acc}')
    
    y_pred = model.predict(x_test)
    print(f'PCA {pca_list[i]} pred acc :', accuracy_score(y_test, y_pred))