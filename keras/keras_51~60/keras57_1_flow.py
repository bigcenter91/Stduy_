from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=5,
    zoom_range=0.1, 
    shear_range=0.7,
    fill_mode='nearest',
)

augment_size = 100

print(x_train.shape) # 60000, 28, 28
print(x_train[0].shape)
print(x_train[1].shape)
print(x_train[0][0].shape)

x = np.tile(x_train[0].reshape(28,28), augment_size).reshape(-1, 28, 28, 1)
print(x.shape)
x = (np.tile(x_train[0], augment_size))
print(x.shape)

print(np.zeros(augment_size))
print(np.zeros(augment_size).shape) # (100, )

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28),
            augment_size).reshape(-1, 28, 28, 1), # x데이터
    np.zeros(augment_size), # y 데이터: 그림만 그릴꺼라 필요없어서 걍 0 넣줬어
    batch_size=augment_size,
    shuffle=True
)

print(x_data)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001A490454D30>

print(x_data[0])
print(x_data[0][0].shape) # (100, 28, 28, 1)
print(x_data[0][1].shape) # y가 들어가 있겠지? (100, )

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap='gray')
    # 총 49개의 plot을 만들거고
plt.show()