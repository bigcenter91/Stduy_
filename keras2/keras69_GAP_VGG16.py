from tensorflow.keras.applications import VGG16

model = VGG16()

model.summary()
# conv 13개와 dense 12개?

print(model.weights)
# 그래서 weight 저장하라는거야