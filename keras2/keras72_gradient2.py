import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 - 4*x + 6
# def f(x):
#     return x**2 -4*x+6
# 위 식과 같다

gradient = lambda x : 2*x -4

x = -10.0       # 초기값
epochs = 20
learning_rate = 0.25

# w = w - lr *△L/△W
print("epoch\t x\t f(x)")
print("{:02d}\t {:6.5f}\t {:6.5f}\t ".format(0, x, f(x)))


for i in range(epochs):
    x = x - learning_rate * gradient(x)

    # print(i+1, '\t' x, '\t' f(x))
    print("{:02d}\t {:6.5f}\t {:6.5f}\t ".format(i+1, x, f(x)))
    
    
# plt.plot(x, y, 'k-')
# plt.plot(2, 2, 'sk')
# plt.grid()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
