import numpy as np
import matplotlib.pyplot as plt

def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

x = np.arange(-5, 5, 0.1)
y = prelu(x)

plt.plot(x, y)
plt.grid()
plt.show()