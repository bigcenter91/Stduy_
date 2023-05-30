import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16()   # include_top = True, input_shape=(224, 224, 3) 디폴트 형태 // cifar100을 쓸려면 224로 증폭해야한다
vgg16 = VGG16(weights='imagenet', include_top=False,    # True로 하면 에러
              input_shape=(32, 32, 3))

vgg16.trainable = False # 디폴트 트루 // 가중치 동결

model=Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# vgg16.trainable = True

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))


# =================================================================
#  vgg16 (Functional)          (None, 1, 1, 512)         14714688   // 위에 있는 vgg16 그냥 쓰겠다는거야

#  flatten (Flatten)           (None, 512)               0

#  dense (Dense)               (None, 10)                5130

# =================================================================
# Total params: 14,719,818
# Trainable params: 14,719,818
# Non-trainable params: 0
# _________________________________________________________________

# trainable: True // False
#             30      30    
#             30       0

# 결국엔 다 해봐

################### 2번 소스에서 아래만 추가 ###################
print(model.layers)

import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
print(layers)
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)

# 0  <keras.engine.functional.Functional object at 0x00000117913573A0>  vgg16      False  // vgg16은 동결이라는 얘기지
# 1  <keras.layers.core.flatten.Flatten object at 0x00000117EB061880>   flatten    True
# 2  <keras.layers.core.dense.Dense object at 0x00000117EB15CD60>       dense      True
# 3  <keras.layers.core.dense.Dense object at 0x0000011791355EE0>       dense_1    True