import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


path = 'd:/study_data/_data/men_women/'
save_path = 'd:/study_data/_save/men_women/'


#1. 데이터

datasets = ImageDataGenerator(
    rescale=1./255,
)

xy_data = datasets.flow_from_directory(
    'd:/study_data/_data/men_women/',
    target_size=(100, 100),
    batch_size=1024,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

mw_x = xy_data[0][0]
mw_y = xy_data[0][1]

print(mw_x.shape, mw_y.shape) # (1024, 100, 100, 3) (1024,)

mw_x_train, mw_x_test, mw_y_train, mw_y_test = train_test_split(
    mw_x, mw_y, train_size=0.7, shuffle=True, random_state=123
)

np.save(save_path + 'keras56_7_mw_x_train.npy', arr=mw_x_train)
np.save(save_path + 'keras56_7_mw_x_test.npy', arr=mw_x_test)
np.save(save_path + 'keras56_7_mw_y_train.npy', arr=mw_y_train)
np.save(save_path + 'keras56_7_mw_y_test.npy', arr=mw_y_test)
