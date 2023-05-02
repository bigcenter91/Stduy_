import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score, mean_squared_error, f1_score 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,	MinMaxScaler, MaxAbsScaler, 	RobustScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs


#1. 데이터

path = "d:/study_data/_data/fac/"
path_save = "d:/study_data/_save/fac/"

train_csv=pd.read_csv(path + 'train_data.csv',index_col=0)
print(train_csv)
print(train_csv.shape) #(2463, 7)

test_csv=pd.read_csv(path + 'test_data.csv',index_col=0)
print(test_csv) 
print(test_csv.shape) #(7389, 7)

#===================================================================================================================
print(train_csv.columns)
# Index(['air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm',
#        'motor_temp', 'motor_vibe', 'type'],
#       dtype='object')
print('train_csv.describe() : ',train_csv.describe())
print('type(train_csv) : ',type(train_csv)) # <class 'pandas.core.frame.DataFrame'>

####################################### 결측치 처리 #######################################  
print('결측치 숫자 : ',train_csv.isnull().sum())  # 없음
#####################train_csv 데이터에서 x와 y를 분리#######################
print('test_csv[type]의 라벨 값 :',np.unique(test_csv['type'])) #[0 1 2 3 4 5 6 7]

x = train_csv.copy()
scaler = StandardScaler()
x = scaler.fit_transform(x)

pca = PCA(n_components=2)
pca_x = pca.fit_transform(x)

kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_x)

centers = kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(3):
    ax.scatter(pca_x[kmeans.labels_ == i, 0], pca_x[kmeans.labels_ == i, 1], label=f"Cluster {i+1}")
ax.scatter(centers[:, 0], centers[:, 1], s=100, c='black', label='Centroids')
ax.set_title('KMeans Clustering')
ax.legend()
plt.show()

train_clusters = kmeans.labels_
print(train_clusters)

train_csv['cluster'] = train_clusters
print(train_csv)


test_x = scaler.transform(test_csv)
test_pca_x = pca.transform(test_x)
test_clusters = kmeans.predict(test_pca_x)


macro_f1_score = f1_score(test_csv['type'], test_clusters, average='macro')
print("macro f1 score: ", macro_f1_score)

submission = pd.read_csv(path +'answer_sample.csv',index_col='type') # index_col을 'type'으로 변경
submission['label'] = test_clusters
submission.to_csv(path_save+'submit0403_fac_03.csv')