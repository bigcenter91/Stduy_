# 맹그러!!
# 가중치 동결과 동결하지 않았을 때 성능비교

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

#1. 데이터
input1 = Input(shape=(32, 32, 3))
vgg16 = VGG16(include_top=False, weights='imagenet')(input1)
# gap1 = GlobalAveragePooling2D()(vgg16)
flt1 = Flatten()(vgg16)
output1 = Dense(10, activation='softmax')(flt1)

model = Model(inputs=input1, outputs=output1)

model.summary()

# GAP로 했을 때 연산량
# vgg16 (Functional)          (None, None, None, 512)   14714688
#  global_average_pooling2d (G  (None, 512)              0
#  lobalAveragePooling2D)
#  dense (Dense)               (None, 10)                5130

# =================================================================
# Total params: 14,719,818
# Trainable params: 14,719,818
# Non-trainable params: 0
# _________________________________________________________________