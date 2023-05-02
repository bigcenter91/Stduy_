import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA #분해 #1.비지도, 2.전처리
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#pca로 차원축소(컬럼압축)을 할 때 필요없는 데이터를 압축함으로써 성능이 좋아 질 수도 있음. (y를 압축하지 않음)

#1. 데이터

datasets = load_diabetes()

print(datasets.feature_names) #판다스에서는 colunms
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets['data']
y = datasets.target

#print(x.shape, y.shape) #(442, 10) (442,)

pca = PCA(n_components= 9)
x = pca.fit_transform(x)
print(x.shape) #(442, 5)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, random_state= 123, train_size=0.8 #디폴트
)

#df = pd.DataFrame(x, columns=datasets.feature_names) #컬럼이름을 넣는 과정
#print(df)

#2. 모델

model = RandomForestRegressor(random_state=123)

#3. 훈련
model.fit(x_train,y_train)

#4. 결과
results = model.score(x_test, y_test)
print("결과는  :", results)

#결과는  : 0.42816745372688414


'''
df['target(y)'] = y
print(df)

print("======================================상관계수 히트 맵 짜잔 =================================================")
print(df.corr())

import matplotlib.pyplot as plt
import seaborn as sns 

sns.heatmap(data=df.corr(), 
            square=True, 
            annot=True,
            cbar=True)

plt.show()
'''