import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = 'd:/study_data/_data/brain/'
save_path = 'd:/study_data/_save/brain/'

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, # 수평
    vertical_flip=True, 
    width_shift_range=0.1, # 10프로만큼의 좌우로 이동한다
    height_shift_range=0.1, #데이터 증폭하는 내용이겠지?
    rotation_range=5,
    zoom_range=1.2, # 약 20프로 확대하겠다
    shear_range=0.7,
    fill_mode='nearest',
    # 증폭 옵션이다
)

train_datagen2 = ImageDataGenerator(
    rescale=1./1,

)

augment_size = 1000

# minmaxscaler 할 필요가 없다 1./255, 하니까 한마디로 0~1사이로 정규화를 한다는 얘기

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary', # 바이너리 0과 1로 : 수치화해서 만들어준다
    color_mode='grayscale', # 흑백
    shuffle=True,
    # 한명이 데이터를 수집한게 아닐 수 있어 사이즈를 동일하게 해준다
    # 이미지 데이터는 데이터의 상위 폴더까지만 지정해준다
) # Found 160 images belonging to 2 classes. x = 160, 200, 200, 1 / y = 160, 


xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(200,200),
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale', 
    shuffle=True,
)

brain_x_train = xy_train[0][0]
brain_y_train = xy_train[0][1]
brain_x_test = xy_test[0][0]
brain_y_test = xy_test[0][1]

print(brain_x_train.shape, brain_x_test.shape) # 5, 200, 200, 1) (5, 200, 200, 1)
print(brain_x_train[0].shape)
print(brain_x_train[0][1].shape)


'''
np.save(save_path + 'keras58_brain_x_train.npy', arr=brain_x_train)
np.save(save_path + 'keras58_brain_y_train.npy', arr=brain_y_train)
np.save(save_path + 'keras58_brain_x_test.npy', arr=brain_x_test)
np.save(save_path + 'keras58_brain_y_test.npy', arr=brain_y_test)
'''