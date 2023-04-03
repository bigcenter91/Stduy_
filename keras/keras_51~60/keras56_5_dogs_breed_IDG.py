# 카테고리컬 잡고 원핫대신에 카테고리컬

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/breed/'
save_path = 'd:/study_data/_save/breed/'

#1. 데이터

datasets = ImageDataGenerator(
    rescale=1./255,
)
# 테스트 데이터는 증폭 하지않는다 예측 할려고 하기때문에 수정하지 않는다

xy_data = datasets.flow_from_directory(
    'd:/study_data/_data/breed/',
    target_size=(500,500), # 모든 부분이 핵심 데이터가 아니기 때문에 잘라야하고 카메라가 또 다 다르기 때문에 잘라줘야한다
    batch_size=1000,
    class_mode='categorical',
    color_mode='rgba',
    shuffle=True,
)
print(xy_data)
#Found 1030 images belonging to 5 classes.
dog_x = xy_data[0][0]
dog_y = xy_data[0][1]

print(dog_x.shape, dog_y.shape)


# print(dog_x_train.shape, dog_x_test.shape) # (700, 500, 500, 4) (300, 500, 500, 4)
# print(dog_y_train.shape, dog_y_test.shape) # (700, 5) (300, 5)


np.save(save_path + 'keras56_5_dog_x.npy', arr=dog_x)
np.save(save_path + 'keras56_5_dog_y.npy', arr=dog_y)

