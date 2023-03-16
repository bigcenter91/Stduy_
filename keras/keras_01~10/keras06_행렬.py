import numpy as np

x1= np.array([[1,2], [3,4]])
x2= np.array([[[1,2,3]]])
x3= np.array([[[1,2,3], [4,5,6]]])
x4= np.array([[1], [2], [3]])
x5= np.array([[[1]], [[2]], [[3]]])
x6= np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
x7= np.array([[[1,2]],[[3,4]], [[5,6]], [[7,8]]])

print(x1.shape) # (2, 2)
print(x2.shape) # (1, 1, 3)
print(x3.shape) # (1, 2, 3)
print(x4.shape) # (3, 1)
print(x5.shape) # (3, 1, 1)
print(x6.shape) # (2, 2, 2)
print(x7.shape) # (4, 1, 2)

# 행, 열 크기 같아야한다