from sklearn.preprocessing import StandardScaler


datasets = load_boston()
x = datasets.data
y = datasets['target']

# Standardization 평균 0 / 분산 1
scaler = StandardScaler()   

scaler = scaler.fit_transform(x)

# 교차검증시
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)