from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # conv2d CNN 하겠다는거다 , Flatten 펼치다

model = Sequential()

model.add(Conv2D(7, (2,2), #보통 cnn에서는 이미지가 커도 2,2 / 3,3 많이 쓴다
                 padding='same',
                 input_shape=(8,8,1))) # 출력 : (N, 7, 7, 7) > (N, 8, 8, 7)_same을 썼을 때
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
# 파라미터 수 = (필터의 높이 * 필터의 너비 * 입력 채널 수 + 편향) * 출력 채널 수(필터스)


#인공지능은 원래 주술적인측면이 강하다

#shape 유지하고 싶으면 'same' 아니면 'valid'
#커널 사이즈가 너무 커지면 데이터 정확?하기 어렵다 