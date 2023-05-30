import pandas as pd
import time
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D
from keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

model_list = [VGG16,
            #   VGG19, 
            #   ResNet50,
            #   ResNet50V2,
            #   ResNet101,
            #   ResNet101V2,
            #   ResNet152,
            #   ResNet152V2,
            #   DenseNet201,
            #   DenseNet121,
            #   DenseNet169,
            #   InceptionV3,
            #   InceptionResNetV2,
            #   MobileNet,
            #   MobileNetV2,
            #   MobileNetV3Small,
            #   MobileNetV3Large,
            #   NASNetMobile,
            #   NASNetLarge,
            #   EfficientNetB0,
            #   EfficientNetB1,
            #   EfficientNetB7,
            #   Xception,
              ]

def m1():
    cat_path = 'c:/_study/_data/_cat_dog/Cat/0.jpg'
    dog_path = 'c:/_study/_data/_cat_dog/Dog/0.jpg'

    cat_dog_path_list = {'cat': cat_path,
                        'dog' : dog_path}
    
    for p in cat_dog_path_list:
        if p in 'cat':
            x_train = load_img(cat_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(cat_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        elif p in 'dog':
            x_train = load_img(dog_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(dog_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)

            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        for m in model_list:
            base_model = m(include_top = False, weights = 'imagenet', input_shape = (32, 32, 3))

            input1 = Input(shape = (32, 32, 3), name = 'hidden1')
            x = base_model(input1)
            gap = GlobalAveragePooling2D(name = 'GAP')(x)
            dense1 = Dense(2, name = 'hidden2')(gap)
            output1 = Dense(2, activation='softmax', name='hidden3')(dense1)

            model = Model(inputs=input1, outputs=output1)

            # For fine-tuning, set trainable of base_model to True
            base_model.trainable = True

            pd.set_option('max_colwidth', -1)

            layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
            results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
            print(results)

            learning_rate = 0.00001
            optimizer = Adam(learning_rate = learning_rate)

            model.compile(loss = 'categorical_crossentropy',
                        optimizer = optimizer,
                        metrics = ['accuracy'])

            es = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            mode = 'min',
                            verbose = 1)

            rl = ReduceLROnPlateau(monitor = 'val_loss',
                                patience = 7,
                                mode = 'auto',
                                verbose = 1)

            s_time = time.time()
            model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    callbacks = [es, rl])
            e_time = time.time()

            loss, acc= model.evaluate(x_test, y_test)
            print(f'고양이, 강아지 : {p}, 모델이름 : {m.__name__}, loss : {loss}, acc : {acc}, 걸린 시간 : {e_time - s_time}')
            print()

m1()

def m2():
    horses_path = 'c:/_study/_data/_horse_or_human/horses/horse01-0.png'
    humans_path = 'c:/_study/_data/_horse_or_human/humans/human01-00.png'

    horses_humans_path_list = {'horses': horses_path,
                               'humans' : humans_path}
    
    for p in horses_humans_path_list:
        if p in 'horses':
            x_train = load_img(horses_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(horses_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        elif p in 'humans':
            x_train = load_img(humans_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(humans_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        for m in model_list:
            base_model = m(include_top = False, weights = 'imagenet', input_shape = (32, 32, 3))

            input1 = Input(shape = (32, 32, 3), name = 'hidden1')
            x = base_model(input1)
            gap = GlobalAveragePooling2D(name = 'GAP')(x)
            dense1 = Dense(2, name = 'hidden2')(gap)
            output1 = Dense(2, activation='softmax', name='hidden3')(dense1)

            model = Model(inputs=input1, outputs=output1)

            # For fine-tuning, set trainable of base_model to True
            base_model.trainable = True

            pd.set_option('max_colwidth', -1)

            layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
            results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
            print(results)

            learning_rate = 0.00001
            optimizer = Adam(learning_rate = learning_rate)

            model.compile(loss = 'categorical_crossentropy',
                        optimizer = optimizer,
                        metrics = ['accuracy'])

            es = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            mode = 'min',
                            verbose = 1)

            rl = ReduceLROnPlateau(monitor = 'val_loss',
                                patience = 7,
                                mode = 'auto',
                                verbose = 1)

            s_time = time.time()
            model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    callbacks = [es, rl])
            e_time = time.time()

            loss, acc= model.evaluate(x_test, y_test)
            print(f'말, 사람 : {p}, 모델이름 : {m.__name__}, loss : {loss}, acc : {acc}, 걸린 시간 : {e_time - s_time}')
            print()

m2()

def m3():
    paper_path = 'c:/_study/_data/_rps/paper/paper01-000.png'
    rock_path = 'c:/_study/_data/_rps/rock/rock01-000.png'
    scissors_path = 'c:/_study/_data/_rps/scissors/scissors01-000.png'

    paper_rock_scissors_path_list = {'paper': paper_path,
                                     'rock' : rock_path,
                                     'scissors' : scissors_path}
    
    for p in paper_rock_scissors_path_list:
        if p in 'paper':
            x_train = load_img(paper_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(paper_path, target_size = (32, 32))
            y_test = [1, 0]
            
            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.

        elif p in 'rock':
            x_train = load_img(rock_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(rock_path, target_size = (32, 32))
            y_test = [1, 0]
            
            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
            
        elif p in 'scissors':
            x_train = load_img(scissors_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(scissors_path, target_size = (32, 32))
            y_test = [1, 0]
            
            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.

            x_train = x_train / 255.
            x_test = x_test / 255.
            
        for m in model_list:
            base_model = m(include_top = False, weights = 'imagenet', input_shape = (32, 32, 3))

            input1 = Input(shape = (32, 32, 3), name = 'hidden1')
            x = base_model(input1)
            gap = GlobalAveragePooling2D(name = 'GAP')(x)
            dense1 = Dense(2, name = 'hidden2')(gap)
            output1 = Dense(2, activation='softmax', name='hidden3')(dense1)

            model = Model(inputs=input1, outputs=output1)

            # For fine-tuning, set trainable of base_model to True
            base_model.trainable = True

            pd.set_option('max_colwidth', -1)

            layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
            results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
            print(results)

            learning_rate = 0.00001
            optimizer = Adam(learning_rate = learning_rate)

            model.compile(loss = 'categorical_crossentropy',
                        optimizer = optimizer,
                        metrics = ['accuracy'])

            es = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            mode = 'min',
                            verbose = 1)

            rl = ReduceLROnPlateau(monitor = 'val_loss',
                                patience = 7,
                                mode = 'auto',
                                verbose = 1)

            s_time = time.time()
            model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    callbacks = [es, rl])
            e_time = time.time()

            loss, acc= model.evaluate(x_test, y_test)
            print(f'가위, 바위, 보 : {p}, 모델이름 : {m.__name__}, loss : {loss}, acc : {acc}, 걸린 시간 : {e_time - s_time}')
            print()

m3()

def m4():
    men_path = 'c:/_study/_data/_men_women/men/00000001.jpg'
    women_path = 'c:/_study/_data/_men_women/women/00000000.jpg'

    men_women_path_list = {'men': men_path,
                           'women' : women_path}
    
    for p in men_women_path_list:
        if p in 'men':
            x_train = load_img(men_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(men_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        elif p in 'women':
            x_train = load_img(women_path, target_size = (32, 32))
            y_train = [1, 0]
            x_test = load_img(women_path, target_size = (32, 32))
            y_test = [1, 0]

            x_train = img_to_array(x_train)
            x_test = img_to_array(x_test)

            x_train = np.expand_dims(x_train, axis=0)
            y_train = np.array(y_train).reshape(1, -1)
            x_test = np.expand_dims(x_test, axis=0)
            y_test = np.array(y_test).reshape(1, -1)

            x_train = x_train / 255.
            x_test = x_test / 255.
        for m in model_list:
            base_model = m(include_top = False, weights = 'imagenet', input_shape = (32, 32, 3))

            input1 = Input(shape = (32, 32, 3), name = 'hidden1')
            x = base_model(input1)
            gap = GlobalAveragePooling2D(name = 'GAP')(x)
            dense1 = Dense(2, name = 'hidden2')(gap)
            output1 = Dense(2, activation='softmax', name='hidden3')(dense1)

            model = Model(inputs=input1, outputs=output1)

            # For fine-tuning, set trainable of base_model to True
            base_model.trainable = True

            pd.set_option('max_colwidth', -1)

            layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
            results = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])
            print(results)

            learning_rate = 0.00001
            optimizer = Adam(learning_rate = learning_rate)

            model.compile(loss = 'categorical_crossentropy',
                        optimizer = optimizer,
                        metrics = ['accuracy'])

            es = EarlyStopping(monitor = 'val_loss',
                            patience = 5,
                            mode = 'min',
                            verbose = 1)

            rl = ReduceLROnPlateau(monitor = 'val_loss',
                                patience = 7,
                                mode = 'auto',
                                verbose = 1)

            s_time = time.time()
            model.fit(x_train, y_train,
                    epochs = 20,
                    batch_size = 32,
                    callbacks = [es, rl])
            e_time = time.time()

            loss, acc= model.evaluate(x_test, y_test)
            print(f'남자, 여자 : {p}, 모델이름 : {m.__name__}, loss : {loss}, acc : {acc}, 걸린 시간 : {e_time - s_time}')
            print()

m4()
