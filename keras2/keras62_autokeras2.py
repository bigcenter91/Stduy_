import autokeras as ak
print(ak.__version__)                                          # 1.0.20
from keras.datasets import mnist
import time
import tensorflow as tf

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# model = ak.ImageClassifier(max_trials=3,
#                            overwrite=False)                    # True일 경우 모델 탐색을 처음부터 다시해
#                         # 너무 선능이 안좋을 때 True 써!!
#                         # 하지만 보통이상 성능이면 True일 때 더 성능이 향상된다
#                         # 

# # max_trials=1, overwrite=False은 한번 돌려봐야한다
path = './_save/autokeras/'
# best_model.save(path + "keras62_autokeras1.h5")

model = tf.keras.models.load_model(path + "keras62_autokeras1.h5")

s_time = time.time()
model.fit(x_train, y_train, epochs=10, validation_split=0.15)
e_time = time.time()

y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)
print('걸린시간 :', round(e_time - s_time), 2)

best_model = model.export_model()
print(best_model.summary())

path = './_save/autokeras/'
best_model.save(path + "keras62_autokeras1.h5")

# cast는 reshape랑 비슷한 얘기야
