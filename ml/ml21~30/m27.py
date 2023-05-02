import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer,load_digits,load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
#1.데이터
datasets = load_iris()
x = datasets.data
y = datasets.target
scaler = RobustScaler()

#datasets = [(load_iris(), 'Iris')]
            # (load_breast_cancer(), 'Breast Cancer'), 
            # (load_digits(), 'Digits'), 
            # (load_wine(), 'Wine')]

# scaler_list = [MinMaxScaler(),
#                MaxAbsScaler(), 
#                StandardScaler(), 
#                RobustScaler()]

scaler = RobustScaler()

model_list = [XGBClassifier(), 
              DecisionTreeClassifier(),
              RandomForestClassifier(),
              GradientBoostingClassifier()]

x = scaler.fit_transform(x)
for model in model_list :
    model.fit(x,y)
    
    def plot_feature_importances(model):
        n_features = datasets.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), datasets.feature_names)
        plt.xlabel('Feature Importances')
        plt.ylabel('Feature')
        plt.ylim(-1, n_features)
        plt.title(model)
        for j in subplot :
            
            plt.subplot(2, 2, j) # 田 형태
            plot_feature_importances(model)
            plt.show()