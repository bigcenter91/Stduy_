import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16()   # include_top = True, input_shape=(224, 224, 3) 디폴트 형태 // cifar100을 쓸려면 224로 증폭해야한다
model = VGG16(weights='imagenet', include_top=False,    # True로 하면 에러
              input_shape=(32, 32, 3))

model.summary()

print(len(model.weights))               #32 > 26
print(len(model.trainable_weights))     #32 > 26

# 풀리 커넥트 레이어
# 26
# 26 13개의 conv 레이어

# 사전학습은 인풋과 아웃풋을 뺀 나머지를 쓴다

############ include_top = True ############
#1. FC layer 원래꺼쓴다
#2. input_shape=(224, 224, 3) 고정값 > 바꿀 수 없다

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0
#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# ........
#  flatten (Flatten)           (None, 25088)             0
#  fc1 (Dense)                 (None, 4096)              102764544
#  fc2 (Dense)                 (None, 4096)              16781312
#  predictions (Dense)         (None, 1000)              4097000
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________

############ include_top = False ############
#1. FC layer 원래꺼 삭제 > 커스터 마이징!!
#2. input_shape=(32, 32, 3) 고정값 > 바꿀 수 있다 > 커스터마이징

#  input_1 (InputLayer)        [(None, 32, 32, 3)]       0
#  block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

# .....
# 플래튼 하단부분(풀리커넥티드 레이어부분) 삭제됨
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0