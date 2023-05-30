# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

# 넘파이까지 저장

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'c:/study_data/_data/cat_dog/petimages/'
save_path = 'c:/study_data/_save/cat_dog/'

# np.save(save_path + '파일명', arr=)

#1. 데이터
datasets = ImageDataGenerator(
    rescale=1./255,
)


xy_data = datasets.flow_from_directory(
        'c:/study_data/_data/cat_dog/PetImages/',
        target_size=(100, 100),
        batch_size=5000,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True)





print(xy_data)
# Found 25000 images belonging to 2 classes.
# <keras.preprocessing.image.DirectoryIterator object at 0x000001AD793B5220>

# print(train_datagen[0])
# print(len(train_datagen[0]))
# <keras.preprocessing.image.DirectoryIterator object at 0x00000274B6C23490>
# 2

print(xy_data[0][1])
# [0. 1. 0. 1. 1. 0. 1. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 1. 1. 1. 1. 1. 1.
#  1. 1. 1. 0. 0. 1. 1. 0.]

# print(train_datagen[0][0].shape) # (5, 100, 100, 3)
# print(train_datagen[0][1].shape) # (5, )
print(xy_data[0][0].shape) # (5, 100, 100, 3)
print(xy_data[0][1].shape) # (5, )

cat_dog_x = xy_data[0][0]
cat_dog_y = xy_data[0][1]

cat_dog_x_train, cat_dog_x_test, cat_dog_y_train, cat_dog_y_test = train_test_split(
    cat_dog_x, cat_dog_y, train_size=0.7, shuffle=True, random_state=123)

np.save(save_path + 'keras56_cat_dog_x_train.npy', arr=cat_dog_x_train)
np.save(save_path + 'keras56_cat_dog_x_test.npy', arr=cat_dog_x_test)
np.save(save_path + 'keras56_cat_dog_y_train.npy', arr=cat_dog_y_train)
np.save(save_path + 'keras56_cat_dog_y_test.npy', arr=cat_dog_y_test)

# np.save(save_path + 'cat_dog_x_train.npy', arr=xy_data[0][0])
# np.save(save_path + 'cat_dog_y_train.npy', arr=xy_data[0][1])


