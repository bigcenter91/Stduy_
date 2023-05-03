import numpy as np
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[-10, 2, 3, 4, 5, 6, 700, 8, 9, 10, 11, 12, 50],
               [100,200,-30, 400, 500, 600, -70000,
               800, 900, 1000, 210, 420, 350]]
               )

aaa = np.transpose(aaa) # 차원을 2차원 형태로 해야하기 때문에


outliers = EllipticEnvelope(contamination=.1) # 전체 데이터 중에 몇프로를 이상치로 할건지

for i, column in enumerate(aaa):
    outliers.fit(column.reshape(-1,1))
    results = outliers.predict(column.reshape(-1,1))
    outliers_save = np.where(results==-1)
    outliers_values = column[outliers_save]
    

    print(f"{i+1} 번째 컬럼의 이상치 : {','.join(map(str,outliers_values))}\n이상치의 위치 : {','.join(map(str,outliers_save))}")