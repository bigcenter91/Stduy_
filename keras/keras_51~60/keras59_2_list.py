import numpy as np
import pandas as pd

a = [[1,2,3],[4,5,6]]

b = np.array(a)
print(b)

c = [[1,2,3], [4,5]]
print(c)

d = np.array(c)
print(d)
#####################################

e = [[1,2,3], ["바보", "맹구", 5, 6]]
print(e)
# 리스트 안에 각각 다른 자료형 구조를 넣어도 상관없다
f = np.array(e)
print(f)

# 리스트는 크기를 다르게 만들 수 있지
# len으로 크기 확인이 가능하다
# numpy