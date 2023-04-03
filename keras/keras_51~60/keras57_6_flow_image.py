from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
np.random.seed(777)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    # vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=5,
    zoom_range=0.1, 
    shear_range=0.7,
    fill_mode='nearest',
)

augment_size = 40000 # 증폭하고 싶은 사이즈

# randidx = np.random.randint(60000, size = 40000)
randidx = np.random.randint(x_train.shape[0], size=augment_size) # 6만은 x_train의 사이즈
#randidx는 한마디로 리스트 형태로 랜덤하게 4만게

print (randidx) # [46600  6826 26068 ... 24633 33816 28132] 랜덤하게 나오지
print (randidx.shape) # 스칼라가 4만, 벡터가 1 (40000,)
print(np.min(randidx), np.max(randidx)) # 1 59992
#randint 4만개 들어갔어도 몇부터 몇까지 들어가는지 알면 좋겠지?

x_augmented = x_train[randidx].copy() # (40000, 28, 28,) 아직 reshape 해주기 전이니까
# x_augmented 변수에 4만개의 데이터가 늘어난다
y_augmented = y_train[randidx].copy()

print(x_augmented)
print(x_augmented.shape, y_augmented.shape) # (40000, 28, 28) (40000,)

# 증폭할려면 4차원 되야겠지 전부다 4차원으로 바꾸자
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

# 이제 변환을 시킵시다
# train_datagen을 flow에 집어넣어서 변환을 해주면 되겠지
# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented, batch_size=augment_size, shuffle=False,
# )

print(x_augmented)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x0000018111193B50>
# 지금은 이터레이터 형태지?
# 원하는건 40000, 28, 28, 1이지?

# 통배치니까
print(x_augmented[0][0].shape) # (40000, 28, 28, 1)

# next를 붙힐려면 어떻게 해야되겠어?
# .next 했을 때 처음에 튜플이 나온다
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size=augment_size, shuffle=False
).next()[0]
print(x_augmented)
print(x_augmented.shape) # (40000, 28, 28, 1)

# x_train과 x_augment를 합체하면 되겠죠?
# x_train = x_train + x_augmented
# print(x_train.shape)

print(np.max(x_train), np.min(x_train)) # 255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented)) 
y_train = np.concatenate((y_train, y_augmented))
x_test = x_test/255.
# (())는 파이썬 문법을 공부하면 된다~

print(np.max(x_train), np.min(x_train)) # 1.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) , (100000,)

#[실습]
# x_augmented 10개와 x_train 10개를 비교하는 이미지 출력할 것
# 서브플롯 2,10?

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i+60000], cmap='gray')
    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()