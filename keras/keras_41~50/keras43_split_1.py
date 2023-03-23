import numpy as np

dataset = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps): 
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

# 6번을 반복하겠어요
# i는 카운트 하나씩 올라간다
# if문, 반복문(for)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape) # 6, 5

# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]

# x = bbb[:, :4] 둘다 같은 표현
x = bbb[:, :-1]
y = bbb[:, -1]

print(x)
print(y)

#언어에 종속적이지 않아야한다
