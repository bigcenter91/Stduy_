import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/horse_or_human/'
save_path = 'd:/study_data/_save/horse_or_human/'

#1. 데이터

datasets = ImageDataGenerator(
    rescale=1./255,
)


xy_data = datasets.flow_from_directory(
    'd:/study_data/_data/horse_or_human/',
    target_size=(150, 150),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
)

print (xy_data)
# Found 1027 images belonging to 2 classes.
# <keras.preprocessing.image.DirectoryIterator object at 0x000002B2D6D5FF10>

# print(xy_data[0])
print(len(xy_data[0]))
# <keras.preprocessing.image.DirectoryIterator object at 0x0000026E15F4FF10> 2

print(xy_data[0][0].shape) # (100, 150, 150, 3)

hh_x = xy_data[0][0]
hh_y = xy_data[0][1]
print(hh_x.shape)
print(hh_y.shape)


hh_x_train, hh_x_test, hh_y_train, hh_y_test = train_test_split(
    hh_x, hh_y, train_size=0.7, shuffle=True, random_state=123,
)

np.save(save_path + '5_hh_x_train', arr=hh_x_train)
np.save(save_path + '5_hh_x_test', arr=hh_x_test)
np.save(save_path + '5_hh_y_train', arr=hh_y_train)
np.save(save_path + '5_hh_y_test', arr=hh_y_test)
