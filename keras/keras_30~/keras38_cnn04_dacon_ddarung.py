from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split


#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)

print(train_csv) 
print(train_csv.shape) # 1459, 10

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)

print(train_csv.shape)


print(test_csv)
print(test_csv.shape) #(715, 9)


print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())
print(train_csv.shape) # (1328, 10)

#2. 모델 구성
print(type(train_csv))
x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=333
)

model = Sequential()

model.add(Conv2D(3, (2,2), 
                  padding='same', input_shape=(5,2,1))) 
                                
model.add(Conv2D(filters=4, 
                  padding='same',
                 kernel_size=(2,2),
                 activation='relu')) 

model.add(Conv2D(10, (2,2),
                 padding='same',)) 
                            
model.add(Flatten())         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='relu'))      
model.add(Dense(1, activation='linear'))
model.summary()



