from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')
# model = ResNet50(weights= None)
# model = ResNet50(weights='경로')


# path = 'C:\study_data\_data\cat_dog\PetImages\Dog\\6.jpg'
path = 'C:\study_data\_data\\my.png'
img = image.load_img(path, target_size=(224, 224))
print(img) # <PIL.Image.Image image mode=RGB size=224x224 at 0x1CF96AFB8B0>

x = image.img_to_array(img)
print("============================image.img_to_array(img)============================")
print(x, '\n', x.shape) #  (224, 224, 3)
print(np.min(x), np.max(x)) # 0.0 255.0

x = x.reshape(x.shape[0], x.shape[1], x.shape[2])
print(x.shape)      # (1, 224, 224, 3)

x = np.expand_dims(x, axis=0)
print(x.shape) # (1, 1, 224, 224, 3)

# 0~1 사이로 바뀌는 정규화
# -1~1 사이는 0에서 중심점 잡는 scaling
# 이미지 위 두개 쓰는데 뭐가 좋다그게 없어

############ -155에서 155 사이로 정규화 ############
print("============================image.img_to_array(img)============================")
x = preprocess_input(x)
print(x.shape) # (1, 1, 224, 224, 3)
print(np.min(x), np.max(x)) # -123.68 151.061

print("======================= model.predict(x) ====================")
x_pred = model.predict(x)
print(x_pred, '\n', x_pred.shape)

print("결과는 :", decode_predictions(x_pred, top=5)[0])
