import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# imputer = IterativeImputer(estimator=XGBRegressor())
# imputer = IterativeImputer(estimator=DecisionTreeRegressor())
# imputer = KNNImputer()
imputer = SimpleImputer(strategy='constant') 
# imputer = SimpleImputer()
path = './_data/ddarung/'
ddarung = pd.read_csv(path + 'train.csv', index_col=0)
ddarung = imputer.fit_transform(ddarung)
print(pd.DataFrame(ddarung))
x = pd.DataFrame(ddarung).drop(9, axis=1)
y = pd.DataFrame(ddarung)[9]

model = XGBRegressor()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print('result : ', result)

y_pred = model.predict(x_test)
print('r2 : ', r2_score(y_test, y_pred))




# XGBRegressor
# result :  0.7513131310632166
# r2 :  0.7513131310632166

# DecisionTreeRegressor
# result :  0.7696998840723361
# r2 :  0.7696998840723361

# KNNImputer
# result :  0.7537200400081671
# r2 :  0.7537200400081671

# SimpleImputer
# result :  0.7621947881031212
# r2 :  0.7621947881031212

# SimpleImputer(strategy='constant')
# result :  0.7640490824937157
# r2 :  0.7640490824937157