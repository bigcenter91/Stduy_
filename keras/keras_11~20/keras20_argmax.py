import numpy as np

a = np.array([[1,2,3], [6,4,5], [7,9,2], [3,2,1], [2,3,1]])
print(a)
print(a.shape) # (5, 3) 
print(np.argmax(a)) #가장 높은 수 자리의 값이 나온다 전체 데이터의
print(np.argmax(a, axis=0)) # [2,2,1] 0은 행이다, 행끼리 비교해서
print(np.argmax(a, axis=1)) # [2 0 1 0 1] 1은 열, 그래서 열끼리 비교
print(np.argmax(a, axis=-1)) # -1은 가장 마지막
# 가장 마지막 축, 이건 2차원이니까 가장 마지막 축은
# 그래서 -1을 쓰면 이 데이터는 1과 동일
# 이래도 헷갈리면 shape를 찍어봐

# 3,4차원 쓸일이 거의 없다

# [[1 2 3]
#  [6 4 5]
#  [7 9 2]
#  [3 2 1]
#  [2 3 1]]             