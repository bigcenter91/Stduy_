import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load data and perform scaling
datasets = load_iris()
x = datasets.data
y = datasets.target
scaler = RobustScaler()
x = scaler.fit_transform(x)

# Define models and subplot positions
model_list = [XGBClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]
subplot_positions = [1, 2, 3, 4]

# Define function to plot feature importances
def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)
    plt.title(type(model).__name__)

# Plot feature importances for each model
fig = plt.figure(figsize=(10, 8))
for i, model in enumerate(model_list):
    model.fit(x, y)  # fit the model before plotting feature importances
    ax = fig.add_subplot(2, 2, subplot_positions[i])
    plot_feature_importances(model)
plt.tight_layout()
plt.show()
