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

xy_data = datasets.flow_from_directory(
    'd:/study_data/_data/breed/',
    target_size=(500,500),
    batch_size=1000,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)
print(xy_data)
#Found 1030 images belonging to 5 classes.
dog_x = xy_data[0][0]
dog_y = xy_data[0][1]

print(dog_x.shape, dog_y.shape)

dog_x_train, dog_x_test, dog_y_train, dog_y_test = train_test_split(
    dog_x, dog_y, train_size=0.7, shuffle=True, random_state=123,
)

np.save(save_path + 'keras56_5_dog_x_train.npy', arr=dog_x_train)
np.save(save_path + 'keras56_5_dog_x_test.npy', arr=dog_x_test)
np.save(save_path + 'keras56_5_dog_y_train.npy', arr=dog_y_train)
np.save(save_path + 'keras56_5_dog_y_test.npy', arr=dog_y_test)
