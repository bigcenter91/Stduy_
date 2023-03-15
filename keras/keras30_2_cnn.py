from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # conv2d CNN 하겠다는거다 , Flatten 펼치다

model = Sequential()                   #(N, 3)
model.add(Dense(10, input_shape=(3,))) #(batch_size, input_dim)
model.add(Dense(units=15))             #(출력 (batch_size, units))
model.summary()
#batch_size, input _dim
#units, filters, output : 양수 들어가겠지

# 용어다 모르겠으면 외워라