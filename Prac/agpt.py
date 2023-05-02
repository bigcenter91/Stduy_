import numpy as np
import pandas as pd


#pd->np 변경
datapd = pd.DataFrame([[1,2], [3,4]])

npdata = datapd.to_numpy()
npdata = datapd.values
print(type(npdata)) #<class 'numpy.ndarray'>
print(npdata)
'''
[[1 2]
 [3 4]]
'''

#np->pd 변경
datanp = np.array([[0,0],[0,1],[1,0],[1,1]])

pddata = pd.DataFrame(datanp, columns=['col1', 'col2'])
print(type(pddata)) #<class 'pandas.core.frame.DataFrame'>
print(pddata)
'''
   col1  col2
0     0     0
1     0     1
2     1     0
3     1     1
'''

#list->numpy 변경
datalist = [[0,0],[0,1],[1,0],[1,1]]

npdata = np.array(datalist)
print(type(npdata)) #<class 'numpy.ndarray'>
print(npdata)
'''
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
'''

#list->pandas 변경
datalist = [[0,0],[0,1],[1,0],[1,1]]

pddata = pd.DataFrame(datalist)
print(type(pddata)) #<class 'pandas.core.frame.DataFrame'>
print(pddata)
'''
   0  1
0  0  0
1  0  1
2  1  0
3  1  1
'''
#pd-> list 변경
datapd = pd.DataFrame([[1,2], [3,4]])

datalist = datapd.values.tolist()
datalist = datapd.to_numpy().tolist()
print(type(datalist)) #<class 'list'>
print(datalist)
'''
[[1, 2], [3, 4]]
'''

#np -> list 

numpy_test = np.array([1, 2], [3, 4])

list_test = numpy_test.tolist()

print(numpy_test)