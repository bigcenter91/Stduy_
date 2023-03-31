import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'


#1. 데이터

datasets = ImageDataGenerator(
    rescale=1./255,
)

xy_data = datasets.flow_from_directory(
    'd:/study_data/_data/rps/',
    target_size=(100, 100),
    batch_size=1024,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True,
)

rps_x = xy_data[0][0]
rps_y = xy_data[0][1]

rps_x_train, rps_x_test, rps_y_train, rps_y_test = train_test_split(
    rps_x, rps_y, train_size=0.7, shuffle=True, random_state=123
)

np.save(save_path + 'keras56_7_rps_x_train.npy', arr=rps_x_train)
np.save(save_path + 'keras56_7_rps_x_test.npy', arr=rps_x_test)
np.save(save_path + 'keras56_7_rps_y_train.npy', arr=rps_y_train)
np.save(save_path + 'keras56_7_rps_y_test.npy', arr=rps_y_test)
