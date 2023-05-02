import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA #분해 #1.비지도, 2.전처리
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#pca로 차원축소(컬럼압축)을 할 때 필요없는 데이터를 압축함으로써 성능이 좋아 질 수도 있음. (y를 압축하지 않음)

#1. 데이터

datasets = load_boston()

x = datasets['data']
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)
for i in range(13, 0, -1):
    pca = PCA(n_components=i)
    x = pca.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=True, random_state=123, train_size=0.8
    )
    # 2. 모델
    model = RandomForestRegressor(random_state=123)

    # 3. 훈련
    model.fit(x_train, y_train)

    # 4. 결과
    results = model.score(x_test, y_test)
    print("n_components={}: 결과는 {}".format(i, results))
    
# n_components=13: 결과는 0.7739282001034407
# n_components=12: 결과는 0.7671011241790612
# n_components=11: 결과는 0.7987180214590718
# n_components=10: 결과는 0.7628652300371246
# n_components=9: 결과는 0.7555345991532362
# n_components=8: 결과는 0.739779769126428
# n_components=7: 결과는 0.7276614366551009
# n_components=6: 결과는 0.7272187337084339
# n_components=5: 결과는 0.5665298521857083
# n_components=4: 결과는 0.39959198512795546
# n_components=3: 결과는 0.3798743171448129
# n_components=2: 결과는 0.3269562273549145
# n_components=1: 결과는 0.1462306760729869