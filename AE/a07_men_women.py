# [실습] keras56_4 남자, 여자의 노이즈를 넣는다 
# predict : 기미 주근깨 제거
# 5개 사진 출력 / 원본 노이즈 아웃풋

from enum import auto
from keras.layers import Dense, Input
from keras.datasets import mnist
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from enum import auto
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D
from keras.datasets import mnist
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten, MaxPooling2D
from sklearn.metrics import r2_score,accuracy_score


datagen = ImageDataGenerator(
    rescale=1./255)
'''
xy = datagen.flow_from_directory(
    'c:\study_data\_data\men_women', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

kim_test = datagen.flow_from_directory('c:\study_data\_data\my',
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )   



np.save('c:\study_data\_save\men_women/kears_56_07_01_x.npy', arr=xy[0][0])
np.save('c:\study_data\_save\men_women/kears_56_07_02_y.npy', arr=xy[0][1])
np.save('c:\study_data\_save\men_women/kears_56_10_kim.npy', arr=kim_test[0][0])
'''

x = np.load('c:\study_data\_save\men_women/kears_56_07_01_x.npy.')
y = np.load('c:\study_data\_save\men_women/kears_56_07_02_y.npy')
kim = np.load('c:\study_data\_save\men_women/kears_56_10_kim.npy')

print(x[0][0].shape)
print(kim[0][0].shape) # (150, 3)
print(kim[0][1].shape) # (150, 3)
print(kim.shape)

x_train,x_test,y_train,y_test=train_test_split(x, y, train_size=0.7, shuffle=True, random_state=337)

print(x_train.shape)
print(x_test.shape)


# x_train = x_train.reshape(4632, 150, 150, 3)
# x_test = x_test.reshape(993, 150, 150, 3)


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 정규분표형태로
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)    
kim_noised = kim + np.random.normal(0, 0.1, size=kim.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 0이하는 0, 1이상은 1로 
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1) 
kim_noised = np.clip(kim_noised, a_min=0, a_max=1) 

print(x_train.shape) # (4632, 150, 150, 3)
print(x_test.shape)  # (1986, 150, 150, 3)



def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(2, 2), strides=2, padding='same', input_shape=(150, 150, 3), activation='relu'))
    model.add(Conv2D(3, 2, padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, 2, padding='same', activation='sigmoid'))
    model.summary()
    return model
# def autoencoder(hidden_layer_size) :
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape=(150 * 150 * 3,), activation='relu'))
#     model.add(Dense(units=150 * 150 * 3, activation='sigmoid'))
#     return model
#
# model = autoencoder(hidden_layer_size=64)
model = autoencoder(hidden_layer_size=154) 
# model = autoencoder(hidden_layer_size=331)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=10) # 노이즈가 없는걸 x 노이즈가 없는걸 y

output = model.predict(x_test_noised)
kim_output = model.predict(kim_noised)


import random



fig, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16, ax17, ax18)) = plt.subplots(3, 6, figsize=(20, 8))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 6)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]) :
    ax.imshow(x_test[random_images[i]].reshape(150, 150, 3))
    if i == 0 :
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
ax.imshow(kim.reshape(150, 150, 3))
    
    
# 노이즈가 들어간 이미지     
for i, ax in enumerate([ax7, ax8, ax9, ax10, ax11, ax12]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(150, 150, 3))
    if i == 0 :
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])    
ax.imshow(kim_noised.reshape(150, 150, 3))

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax13, ax14, ax15, ax16, ax17, ax18]) :
    ax.imshow(output[random_images[i]].reshape(150, 150, 3))
    if i == 0 :
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
ax.imshow(kim_output.reshape(150, 150, 3))
   
plt.tight_layout()    
plt.show()