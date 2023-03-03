import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x =np.array([1,2,3,4,5])
y =np.array([4,5,6,7,8])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y
    test_size=0.3
    random
)
