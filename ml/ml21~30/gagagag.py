import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_digits,load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# datasets
datasets = [(load_iris(), 'Iris')]
            # (load_breast_cancer(), 'Breast Cancer'), 
            # (load_digits(), 'Digits'), 
            # (load_wine(), 'Wine')]

# scaler
scaler = RobustScaler()

# models
model_list = [XGBClassifier(), 
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              GradientBoostingClassifier()]

for dataset, name in datasets:
    x, y = dataset['data'], dataset['target']  # unpack dataset object from tuple
    x = scaler.fit_transform(x)
    
    for i, model in enumerate(model_list):
        model.fit(x, y)
        
        def plot_feature_importances(model, name, subplot):
            n_features = x.shape[1]
            plt.barh(np.arange(n_features), model.feature_importances_, align='center')
            plt.yticks(np.arange(n_features), dataset['feature_names'])
            plt.xlabel('Feature Importances')
            plt.ylabel('Feature')
            plt.ylim(-1, n_features)
            plt.title(name + ' - ' + str(model.__class__.__name__))
            plt.subplot(8, 8, subplot) # 田 형태
            plt.show()


        
