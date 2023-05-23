# [실습] 얼리스타핑 적용하려면 어떻게 면 될까요?
# 1. 최소값을 넣을 변수 하나, 카운트할 변수 하나 준비!!!
# 2. 다음 에포에 값과 최소값이 갱신되면 그 변수에 최소값을 넣어주고,
#    그 변수에 최소값을 넣어주고, 카운트 변수 초기화
# 3. 갱신이 안되면 카운트 변수 ++1
#    카운트 변수가 내가 원하는 얼리스타핑 갯수에 도달하면 for문을 stop

x = 10
y = 10
w = 10
lr = 0.004
epochs = 10000
count = 0
min_loss = 0
earlystopping_count = 3

loss_list = []
for i in range(epochs):
    hypothesis = x * w
    loss = (hypothesis - y) ** 2            # mse
    
    print('epoch : ', i+1, 'loss : ', loss, '\tPredict : ', hypothesis)
    
    up_predict = x * (w+lr)
    up_loss = (y - up_predict) ** 2 
    
    down_predict = x * (w-lr)
    down_loss = (y - down_predict) ** 2
    
    if (up_loss > down_loss):
        w = w - lr
    elif (up_loss < down_loss):
        w = w + lr
        
    loss_list.append(loss)

    if count != 0:
        if min_loss < loss_list[i]:
            count +=1
        elif min_loss > loss_list[i]:
            count = 0
    if i > 0 and loss_list[i-1] < loss_list[i] and count==0:
        min_loss = loss_list[i-1]
        count+=1
    if count == earlystopping_count:
        break
