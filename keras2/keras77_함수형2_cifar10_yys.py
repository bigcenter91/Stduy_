import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score


base_model = VGG16(weights = 'imagenet', include_top=False,
                   input_shape=(32, 32, 3))
# print(base_model.output) # 마지막 레이어라는 뜻이야
# KerasTensor(type_spec=TensorSpec(shape=(None, None, None, 512), 
#                                  dtype=tf.float32, name=None),
#             name='block5_pool/MaxPool:0', description="created by layer 'block5_pool'")

x = base_model.output
x = GlobalAveragePooling2D()(x)
output1 = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output1)

model.summary()
