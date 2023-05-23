import numpy as np
import matplotlib.pyplot as plt

def selu(x, alpha=1.67326, scale=1.0507):
    condition = x > 0
    return scale * np.where(condition, x, alpha * (np.exp(x) - 1))

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()