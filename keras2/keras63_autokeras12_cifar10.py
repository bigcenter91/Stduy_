import autokeras as ak
from keras.datasets import cifar10

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# AutoKeras 분류기 생성
clf = ak.ImageClassifier(max_trials=10)

# 데이터셋으로 분류기를 훈련
clf.fit(x_train, y_train, epochs=10)

# 분류기의 성능 평가
accuracy = clf.evaluate(x_test, y_test)[1]
print('acc', accuracy)