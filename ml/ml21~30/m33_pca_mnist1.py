import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

#1.데이터

#(x_train, y_train), (x_test,y_test) = mnist.load_data()
(x_train, __), (x_test,_) = mnist.load_data() #이렇게도 가능.

#print(__.shape) #메모리 할당이 됨. 변수를 특수문자를 사용해도됨.

#x = np.concatenate((x_train,x_test),axis = 0)
x = np.append(x_train,x_test, axis= 0)
#print(x.shape)

x = x.reshape(70000, -1)
# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# print(x.shape)

# #-> 쫙피면 70000, 784의 컬럼으로 볼 수 있음.

pca = PCA(n_components= 784) #증폭이 안됨.
x = pca.fit_transform(x) #print(x.shape)
pca_EVR = pca.explained_variance_ratio_ #EVR임
cumsum = np.cumsum(pca_EVR)
#print(cumsum)

print(np.argmax(cumsum >= 0.95) + 1) #154 +1을한 이유 0부터 시작하기 위해서
print(np.argmax(cumsum >= 0.99) + 1) #331 +1을한 이유 0부터 시작하기 위해서
print(np.argmax(cumsum >= 0.999) + 1) #486 +1을한 이유 0부터 시작하기 위해서
print(np.argmax(cumsum >= 1.0) + 1) #712 +1을한 이유 0부터 시작하기 위해서

#chat gpt
# cumulative_variance_ratio = np.cumsum(pca_EVR)
# # n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
# # n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
# # n_components_999 = np.argmax(cumulative_variance_ratio >= 0.999) + 1
# # n_components_100 = np.argmax(cumulative_variance_ratio >= 1) + 1

# n_components_95 = np.searchsorted(cumulative_variance_ratio, 0.95, side='right')
# n_components_99 = np.searchsorted(cumulative_variance_ratio, 0.99, side='right')
# n_components_999 = np.searchsorted(cumulative_variance_ratio, 0.999, side='right')
# n_components_100 = np.searchsorted(cumulative_variance_ratio, 1, side='right')

# print("Number of components for 95% variance:", n_components_95)
# print("Number of components for 99% variance:", n_components_99)
# print("Number of components for 99.9% variance:", n_components_999)
# print("Number of components for 100% variance:", n_components_100)
