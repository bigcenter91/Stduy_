import numpy as np
aaa = np.array([-10, 2, 3, 4, 5, 6, 700,
                  8, 9, 10, 11, 12, 50])
aaa = aaa.reshape(-1, 1) # 차원을 2차원 형태로 해야하기 때문에

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.1) # 전체 데이터 중에 몇프로를 이상치로 할건지

outliers.fit(aaa)
results = outliers.predict(aaa)
print(results) # [ 1  1  1  1  1  1 -1  1  1  1  1  1 -1] -1이 이상치라는걸 알 수 있다
