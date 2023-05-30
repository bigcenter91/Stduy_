import numpy as np
import matplotlib.pyplot as plt

def elu(x, alpha=1.0):
    condition = x > 0
    return np.where(condition, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = elu(x)

plt.plot(x, y)
plt.grid()
plt.show()