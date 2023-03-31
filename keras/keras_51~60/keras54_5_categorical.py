import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# minmaxscaler 할 필요가 없다 1./255, 하니까 한마디로 0~1사이로 정규화를 한다는 얘기

test_datagen = ImageDataGenerator(
    rescale=1./255,
    
    # train이랑 다른 이유는 평가 데이터를 증폭한다는건 데이터를 조작한다는거다
)
# 이터레이터 형태로 수치화 한다
#x, y빼서 쓰면 되겠네
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(100,100),
    batch_size=5, # 전체 데이터 쓸려면 160 넣어라 한방에 할려면 그냥 크게 잡아서 하면 된다 전체 데이터 갯수 이상을 늘려라
    class_mode='categorical', # categorical로 하면 원핫인코딩을 해준다 / binary: 01 > cate: 1, 0 / 0, 1
    color_mode='grayscale', # 흑백
    # color_mode='rgb', # grayscale: 흑백 rgb: 칼라 rgba: 칼라에 투명도
    # 흑백 사진이여도 칼라로 하면 채널이 3개라서 3으로 나온다
    shuffle=True,
    # 한명이 데이터를 수집한게 아닐 수 있어 사이즈를 동일하게 해준다
    # 이미지 데이터는 데이터의 상위 폴더까지만 지정해준다
) # Found 160 images belonging to 2 classes. x = 160, 200, 200, 1 / y = 160, 


xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100,100),
    batch_size=5,
    class_mode='categorical', # class mode : y 라벨 갯수
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

print(xy_train[0][0].shape) # (5, 100, 100, 1)
print(xy_train[0][1].shape) # (5, 2)

# 배치를 160으로하면 이미지를 한방에 때려서 하이패스하겠다는 뜻


# 배치사이즈 크기, X,Y

print("================================================")

print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) # <class 'tuple'> 리스트와 튜플의 차이 바꾸지 못한다
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 현재 (5, 200, 200, 1) 짜리 데이터가 32덩어리

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

hist = model.fit(xy_train, epochs=30, # xy_train을 넣으면 x,y 데이터/ 배치 사이즈까지 끝난거다
                    steps_per_epoch=32,  # 전체 데이터 크기/batch = 160/5 = 32 // 32(계산한만큼) 이상주면 에러난다 // 안써줘도 돌아간다
                    validation_data=xy_test,
                    validation_steps=24, # 발리데이터/batch = 120/5 = 24
                    )

# fit_generator 대신에 fit을 쓰면 된다

# numpy로 수치를 저장해놓으면 좋다

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# # print(acc[-1])
# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.subplot(2,1,1)
# plt.plot(loss)
# plt.plot(val_loss)

# plt.subplot(2,1,2)
# plt.plot(acc)
# plt.plot(val_acc)

# plt.show()




#4. 평가, 예측
