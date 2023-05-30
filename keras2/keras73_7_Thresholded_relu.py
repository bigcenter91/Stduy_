import numpy as np
import matplotlib.pyplot as plt

def thresholded_relu(x, theta=1.0):
    return np.where(x > theta, x, 0)

x = np.arange(-5, 5, 0.1)
y = thresholded_relu(x)

plt.plot(x, y)
plt.grid(True)
plt.show()