import numpy as np
import matplotlib.pyplot as plt

# 0보다 크면 그 값을 쓰고 0보다 작으면 버리고?
# def relu(x):
#     return np.maximum(0, x)

relu = lambda x: np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu(x)

plt.plot(x, y)
plt.grid()
plt.show()

# selu, elu, 