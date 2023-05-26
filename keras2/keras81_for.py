import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')

cat_dog = 'C:\study_data\_data\cat_dog/PetImages/'
horse_human = 'C:\study_data\_data\horse_or_human'
rps = 'C:\study_data\_data\rps'
men_women = 'C:\study_data\_data/men_women/'
v = f'C:\study_data\_data\\my.png'

target_x = 100
target_y = 100

img = image.load_img(v, target_size=(target_x, target_y))
img = image.img_to_array(img)/255.
img = np.expand_dims(img, axis=0)

datagen = ImageDataGenerator(rescale=1./255)

path_list = [cat_dog, horse_human, rps, men_women]

model_list = [VGG19, Xception, ResNet50, ResNet101, InceptionV3, InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0]

for k in range(4):
    data = datagen.flow_from_directory(path_list[k], target_size=(target_x, target_y), batch_size=5000, class_mode='categorical', color_mode='rgb', shuffle=True)
    x, y = data[0][0], data[0][1]  
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=337)
    for j in range(len(model_list)):
        try:
            pretrained = model_list[j](weights='imagenet', include_top=False, input_shape=(target_x, target_y, 3))

            result_list = []
            test_acc_list = []

            for i in range(4):
                if 0 <= i < 2:
                    input1 = pretrained
                elif 2 <= i:
                    pretrained.trainable = False
                    input1 = pretrained

                if i == 0 or i == 2:
                    flat = Flatten()(input1.output)
                
                elif i == 1 or i == 3:
                    flat = GlobalAveragePooling2D()(input1.output)
                
                dense2 = Dense(100)(flat)
                dense3 = Dense(100)(dense2) 
                output = Dense(len(np.unique(y_train)), activation='softmax')(dense3)
                
                model = Model(inputs=input1.input, outputs=output)
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
                model.fit(x_train, y_train, epochs=10, batch_size=128)
                result = model.evaluate(x_test, y_test)
                acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(x_test), axis=1))
                
                print(f'{path_list[k]}\n{model_list[j].__name__}\n{i+1} result : {result}\nacc : {acc}')
                result_list.append(result[0])
                test_acc_list.append(acc)
                
                if k == 0:
                    v_pred = np.argmax(model.predict(img), axis=1)
                    if v_pred == 0:
                        print('Cat')
                    else:
                        print('Dog')
            print(f'{path_list[k]}\n{model_list[j].__name__}\n{result_list}\n{test_acc_list}')
        except:
            print(f'{path_list[k]}\n{model_list[j].__name__} encounter an unexpected error')
            continue