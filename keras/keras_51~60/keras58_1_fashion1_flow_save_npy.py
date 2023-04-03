from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size = 10000

randidx = np.random.randint(x_train.shape[0], size=augment_size)
print(randidx.shape) # (10000,)
print(np.min(randidx), np.max(randidx)) # 7 59992

x_augmented = x_train[randidx].copy()
print(x_augmented)
print(x_augmented.shape) # (10000, 28, 28)
y_augmented = y_train[randidx].copy()
print(y_augmented.shape) # (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)


print(x_augmented.shape) # (10000, 28, 28, 1)

print(np.max(x_train), np.min(x_train)) # 255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255.

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) 

print(x_train.shape, y_train.shape) # (70000, 28, 28, 1) (70000, 10)

save_path = 'd:/study_data/_save/fmnist/'
np.save(save_path + 'keras58_fmnist_x_train.npy', arr=x_train)
np.save(save_path + 'keras58_fmnist_y_train.npy', arr=y_train)
np.save(save_path + 'keras58_fmnist_x_test.npy', arr=x_test)
np.save(save_path + 'keras58_fmnist_y_test.npy', arr=y_test)