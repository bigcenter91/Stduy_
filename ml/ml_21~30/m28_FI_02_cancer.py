from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 

#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.2,                                
    shuffle=True, random_state =58525)

print(x.shape,y.shape)
print(datasets.feature_names)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,
                                                    random_state=123,shuffle=True)


#2. 모델 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor

model1 = DecisionTreeClassifier()
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()


#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#4. 예측

from sklearn.metrics import accuracy_score, r2_score

# result = model1.score(x_test,y_test)
# print("model.score:",result)

# y_predict = model1.predict(x_test)
# acc = accuracy_score(y_test,y_predict)

# print( 'accuracy_score :',acc)
# print(model1,':') 
# print("===================================")


# result2 = model2.score(x_test,y_test)
# print("model2.score:",result2)

# y_predict2 = model2.predict(x_test)
# acc2 = accuracy_score(y_test,y_predict2)

# print( 'accuracy2_score :',acc2)
# print(model2,':')    
# print("===================================")


# result3 = model3.score(x_test,y_test)
# print("model3.score:",result3)

# y_predict3 = model3.predict(x_test)
# acc3 = accuracy_score(y_test,y_predict3)

# print( 'accuracy3_score :',acc3)
# print(model3,':')   
# print("===================================")


# result4 = model4.score(x_test,y_test)
# print("model4.score:",result4)

# y_predict4 = model4.predict(x_test)
# acc4 = accuracy_score(y_test,y_predict4)

# print( 'accuracy4_score :',acc4)
# print(model4,':') 
# print("===================================")


model_list = [model1, model2, model3, model4]

for model in model_list:
    result = model.score(x_test,y_test)
    print("model.score:",result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)

    print( 'accuracy_score :',acc)
    print(model,':') 
    print("===================================")
    