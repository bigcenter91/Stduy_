import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor


x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=337,
    shuffle=True,
    test_size=0.2,
    stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {
            'n_estimators' : [100],
            'learning_rate' : [0.5],
            'max_depth' : [2],
            'gamma' : [0],
            'min_child_weight' : [0.5],
            'subsample' : [0.7],
            'colsample_bytree' : [0.7],
            'colsample_bytree1' : [0],
            'colsample_bynode' : [0.2],
            'reg_alpha' : [0.01],
            'reg_lamda' : [0]
}

# 2. 모델
xgb = XGBClassifier(random_state = 337)
model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=-1)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 예측
print('BEST PARAMETERS: ', model.best_params_)
print('BEST SCORE: ', model.best_score_)

results = model.score(x_test, y_test)
print('FINISH SCORE: ', results)