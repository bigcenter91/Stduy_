import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler
from sklearn.utils import all_estimators

warnings.filterwarnings(action='ignore')

data_list = [load_iris(return_X_y=True),
             load_breast_cancer(return_X_y=True),
             load_digits(return_X_y=True),
             load_wine(return_X_y=True)]

data_name_list = ['iris', 
                  'breast_cancer', 
                  'digits', 
                  'wine']

scaler_list = [MinMaxScaler(), StandardScaler(), MaxAbsScaler(), RobustScaler()]

scaler_name_list = ['MinMaxScaler', 
                    'StandardScaler', 
                    'MaxAbsScaler', 
                    'RobustScaler']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

for index, value in enumerate(data_list):
    x, y = value
    print("==================", data_name_list[index], "======================")
    for i, value2 in enumerate(scaler_list):
        scaler = value2
        max_score = 0
        max_name = '최대값'
        max_scaler = 'max'
        for name, algorithm in all_estimators(type_filter='classifier'):
            try:
                model = algorithm()
                scores = cross_val_score(model, scaler.fit_transform(x), y, cv=kfold)
                results = round(np.mean(scores), 4)
                if max_score < results:
                    max_score = results
                    max_name = name
                    max_scaler = scaler_name_list[i]
            except:
                continue
        print('Scaler:', max_scaler, 'Model:', max_name, 'Score:', max_score)
    print("======================================================")
    
# ============== iris ======================
# Scaler: MinMaxScaler Model: LinearDiscriminantAnalysis Score: 0.98
# Scaler: StandardScaler Model: LinearDiscriminantAnalysis Score: 0.98
# Scaler: MaxAbsScaler Model: LinearDiscriminantAnalysis Score: 0.98
# Scaler: RobustScaler Model: LinearDiscriminantAnalysis Score: 0.98
# ================================================
# ============== breast_cancer ======================
# Scaler: MinMaxScaler Model: LinearSVC Score: 0.9736
# Scaler: StandardScaler Model: SGDClassifier Score: 0.9771
# Scaler: MaxAbsScaler Model: LogisticRegressionCV Score: 0.9771
# Scaler: RobustScaler Model: LogisticRegressionCV Score: 0.9789
# ================================================
# ============== digits ======================
# Scaler: MinMaxScaler Model: LabelPropagation Score: 0.9878
# Scaler: StandardScaler Model: SVC Score: 0.98
# Scaler: MaxAbsScaler Model: LabelPropagation Score: 0.9878
# Scaler: RobustScaler Model: ExtraTreesClassifier Score: 0.9783
# ================================================
# ============== wine ======================
# Scaler: MinMaxScaler Model: RidgeClassifier Score: 0.9889
# Scaler: StandardScaler Model: ExtraTreesClassifier Score: 0.9833
# Scaler: MaxAbsScaler Model: RandomForestClassifier Score: 0.9889
# Scaler: RobustScaler Model: ExtraTreesClassifier Score: 0.9889
