import numpy as np

datasets = np.array(range(1, 41)).reshape(10, 4) # 1부터 40까지 벡터/ 1차원
print(datasets)
print(datasets.shape) # (10, 4)

# x_data = datasets(:, :3) #3-1
x_data = datasets[:, :-1]
# x_data = datasets[0:3]
y_data = datasets[:, -1]
print(x_data)
print(y_data)
print(x_data.shape, y_data.shape) # (10, 3) (10,)

timesteps = 3

#####x 만들기 #####
def split_x(dataset, timesteps): 
    aaa = []
    for i in range(len(dataset) - timesteps ):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

# 6번을 반복하겠어요
# i는 카운트 하나씩 올라간다
# if문, 반복문(for)

# timesteps 6으로 해야 1시간 뒤에 걸 맞춘다

bbb = split_x(x_data, timesteps)
print(bbb)
print(bbb.shape) # (5, 5, 3)

##### y만들기 #####
y_data = y_data[timesteps:]
print (y_data)
