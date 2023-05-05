import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer

x, y = make_blobs(random_state=337)

print(x)
print(y)
print(x.shape, y.shape)