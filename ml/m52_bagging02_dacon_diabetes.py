import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier


#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)  
# [652 rows x 9 columns] #(652,9)

test_csv= pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv) 
#(116,8) #outcome제외

# print(train_csv.isnull().sum()) #결측치 없음

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=333, train_size=0.8, stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

model = BaggingClassifier(BaggingClassifier(),
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=333,
                          bootstrap=True,
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('score : ', model.score(x_test, y_test))
print('acc : ', accuracy_score(y_test, y_pred))


# score :  0.6793893129770993 // DecisionTreeClassifier
# acc :  0.6793893129770993

# score :  0.7404580152671756 // RandomForestClassifier
# acc :  0.7404580152671756

# score :  0.732824427480916 // BaggingClassifier
# acc :  0.732824427480916 

