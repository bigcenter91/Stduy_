import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

#1 데이터
path = 'd:/study_data/_data/men_women/'
save_path = 'd:/study_data/_save/men_women/'
# print(x_train.shape,y_train.shape)

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

xy_train = train_datagen.flow_from_directory(path, target_size = (100, 100), batch_size = 30, class_mode = 'binary', color_mode = 'rgb', shuffle = False)
xy_test = train_datagen.flow_from_directory(path, target_size = (100, 100), batch_size = 30, class_mode = 'binary', color_mode = 'rgb', shuffle = False)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]

augment_size = 10000 # 증폭

np.random.seed(0)

randindex = np.random.randint(x_train.shape[0], size = augment_size)
# randindex = np.random.randint(xy_train.shape[0], size = augment_size)
print(randindex) # [35184 43823  4329 ... 31084  8272 16972]
print(randindex.shape) # (40000,)
print(np.min(randindex), np.max(randindex))

x_augmented = x_train[randindex].copy() # 40000개의 데이터가 들어감
y_augmented = y_train[randindex].copy() # 데이터가 중복이 되지않고 증폭이된다.
# print(x_augmented)
# print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# x_train = x_train.reshape(50000, 32, 32, 3)
# x_test = x_test.reshape(x_test.shape[0], # 60000
#                         x_test.shape[1], # 28
#                         x_test.shape[2], 1) # 28, 1
# x_augmented = x_augmented.reshape(x_augmented.shape[0], # 40000
#                                   x_augmented.shape[1], # 28
#                                   x_augmented.shape[2], 1) # 28, 1
print(x_train.shape) # (60000, 28, 28, 1)
print(x_augmented.shape) # (40000, 28, 28, 1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size = augment_size, shuffle = False).next()[0]

x_train = np.concatenate((x_train/255., x_augmented), axis = 0) # 함수를 주어진 순서대로 결합한다.
y_train = np.concatenate((y_train, y_augmented), axis = 0) # 함수를 주어진 순서대로 결합한다.
x_test = x_test/255.

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

np.save(save_path + "keras58_men_women_x_train.npy", arr = x_train)
np.save(save_path + "keras58_men_women_y_train.npy", arr = y_train)
np.save(save_path + "keras58_men_women_x_test.npy", arr = x_test)
np.save(save_path + "keras58_men_women_y_test.npy", arr = y_test)