from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten # conv2d CNN 하겠다는거다 , Flatten 펼치다

model = Sequential()

model.add(Conv2D(7, (2,2), 
                 input_shape=(8,8,1))) # 출력 : (N, 7, 7, 7) 
                            # batch_size, rows, columns, channels = input_shape
                            
                            # 계산:  (2*2*1+1)*7 = 35
                            
model.add(Conv2D(filters=4, #cnn에서는 output을 filters라 한다
                 kernel_size=(3,3),
                 activation='relu')) # relu니까 전부 양수로 던져주겠지, 음수는 다 0으로 되겠지 / (N, 5, 5, 4)

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
# 파라미터 수 = (필터의 높이 * 필터의 너비 * 입력 채널 수 + 편향) * 출력 채널 수


#flatten은 모양만 바꿔준다






# 행 무시, 열 우선 
#5*5를 2*2로 자르면 4*4가 된다
#2,2도 하이퍼파라미터가 되겠지
#5*5*1의 데이터가 2*2로 하면 4*4*7이 된다 = 4*4를 7장으로 늘릴거야
#특성은 모으고 데이터를 늘려서
#가장 자리는 연산을 한번 밖에 안한다