from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # conv2d CNN 하겠다는거다 , Flatten 펼치다

model = Sequential()

model.add(Conv2D(7, (2,2), #보통 cnn에서는 이미지가 커도 2,2 / 3,3 많이 쓴다
                 padding='valid',
                 strides = 2,
                 input_shape=(9,9,1))) # 출력 : (N, 7, 7, 7)
                            # batch_size, rows, columns, channels = input_shape
                            
                            # 계산:  (2*2*1+1)*7 = 35
                            
model.add(Conv2D(filters=4, #cnn에서는 output을 filters라 한다
                 kernel_size=(3,3), # kernel size가 더 클 경우 양쪽 다 들어간다
                 padding='same', # 패딩 디폴트는 valid
                 activation='relu')) # relu니까 전부 양수로 던져주겠지, 음수는 다 0으로 되겠지 / (N, 5, 5, 4) > (None, 8, 8, 4)_same을 썼을 때

                            # 계산: (3*3*7+1)*4 = 256

model.add(Conv2D(10, (2,2))) # (N, 4, 4, 10) 노드 하나에 데이터 하나
                            # 계산: (2*2*4+1)*10 = 170 (입력 채널 수는 output shape 뒷자리)
                            
model.add(Flatten())         # (N, 4*4*10 > N, 160)

model.add(Dense(32, activation='relu'))
                             # (161*32 = 5152) / bias 1 더 해준다

model.add(Dense(10, activation='relu'))
                             # (33*10 = 330)
model.add(Dense(3, activation='softmax'))
model.summary()
                             # (10+1*3 = 33)

# padding = 보폭이 1 디폴트가 1이다
# maxpooling = 디폴트가 2가 되겠지
# 3이면 데이터가 손실 되겠지, 커널 사이즈보단 더 주지않겠지
# 한마디로 커널 사이즈의 보폭
# 5*5를 2*2로 했을 때 남는 데이터는 손실