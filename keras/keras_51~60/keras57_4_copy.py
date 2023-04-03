# 아래와 같이 copy할 경우 문제가 있다

# x_augmented = x_train[randidx]
# y_augmented = y_train[randidx]

import numpy as np
aaa = np.array([1,2,3])
bbb = aaa

print(bbb)

bbb[0] = 4
print(bbb)
print(aaa)

# 처음에 aaa라는 메모리를 생성한다 numpy의 주소값이 공유가 되는거야

print("==========================")

ccc = aaa.copy() # 이렇게 하면 새로운 메모리 구조가 생성되는거야
ccc[1] = 7

print (ccc)
print (aaa)
# copy라는걸 쓰면 안바뀌는걸 확인 할 수 있다
