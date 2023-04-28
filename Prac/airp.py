import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import IncrementalPCA

# 1. 데이터 로드
path = 'c:/study_data/_data/dacon_airplane/'
path_save = 'c:/study_data/_save/dacon_airplane/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# 2. 데이터 전처리
# Combine train and test data for consistent encoding
combined_csv = pd.concat([train_csv, test_csv])

# One-hot encode categorical variables
categorical_cols = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']
combined_encoded = pd.get_dummies(combined_csv, columns=categorical_cols)

# Split train and test data again
train_encoded = combined_encoded.iloc[:len(train_csv)]
test_encoded = combined_encoded.iloc[len(train_csv):]

# 제한된 열 수 선택 (예: 500개) 및 데이터 형식 변경
x = train_encoded.drop(['Cancelled', 'Diverted'], axis=1).iloc[:, :500].astype(np.float32)
y = train_encoded['Delay'].fillna(0).astype(np.int8)
test_csv_encoded = test_encoded.drop(['Cancelled', 'Diverted'], axis=1).iloc[:, :500].astype(np.float32)

# 3. PCA 적용
n_components = 500  # 선택한 주성분 개수로 설정
batch_size = 1000  # 배치 크기 설정
n_batches = int(np.ceil(x.shape[0] / batch_size))

# Incremental PCA를 사용하여 메모리 사용량을 줄입니다.
pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

x_pca = np.zeros((x.shape[0], n_components), dtype=np.float32)  # PCA 적용된 데이터를 저장할 배열 생성

for i in range(n_batches):
    batch_start = i * batch_size
    batch_end = min((i + 1) * batch_size, x.shape[0])
    x_batch = x.iloc[batch_start:batch_end]
    x_pca[batch_start:batch_end] = pca.partial_fit_transform(x_batch)

test_csv_encoded_pca = pca.transform(test_csv_encoded)

# 4. 모델 학습 및 예측
# Create a logistic regression model
model = LogisticRegression()

# Create a self-training classifier and fit it on the labeled data
self_training_model = SelfTrainingClassifier(model, threshold=0.9, max_iter=100)
self_training_model.fit(x_pca, y)

# Predict the labels and probabilities of the unlabeled data
predicted_labels = self_training_model.predict(test_csv_encoded_pca)
predicted_probs = self_training_model.predict_proba(test_csv_encoded_pca)

# 5. 결과 저장 및 제출
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Delayed'] = predicted_probs[:, 1]
submission['Not_Delayed'] = predicted_probs[:, 0]
submission.to_csv(path_save + 'airplane_submission.csv')
