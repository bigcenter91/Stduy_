import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# minmaxscaler 할 필요가 없다 1./255, 하니까 한마디로 0~1사이로 정규화를 한다는 얘기

test_datagen = ImageDataGenerator(
    rescale=1./255,
    
    # train이랑 다른 이유는 평가 데이터를 증폭한다는건 데이터를 조작한다는거다
)
# 이터레이터 형태로 수치화 한다
#x, y빼서 쓰면 되겠네
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(200,200),
    batch_size=7,
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
) # Found 120 images belonging to 2 classes. x = 120, 200, 200, 1 / y = 120, 
# numpy np.unique
# pandas value count??

# <keras.preprocessing.image.DirectoryIterator object at 0x000002AAA9385F70>
# 0x000002AAA9385F70 메모리 주소 값

print(xy_train) 
print(xy_train[0]) # 32
# y : array([0., 0., 1., 0., 1.] / 그 앞에 나온게 x값
print(len(xy_train[0])) # 2
print(xy_train[0][0]) # 0번째에 0번째 / X가 다섯개 들어가있다
print(xy_train[0][1]) # [1. 0. 0. 0. 1.]

print(xy_train[0][0].shape) # (5, 200, 200, 1)
print(xy_train[0][1].shape) # (5,)

# 배치사이즈 크기, X,Y

print("================================================")

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> 리스트와 튜플의 차이 바꾸지 못한다
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

