import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],
                [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])
aaa = np.transpose(aaa)    # 13, 2

print(aaa)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25,50,75])
    print("1사분위 : ", quartile_1)   # 
    print("q2 :", q2)                # 
    print("3사분위 : ", quartile_3)   # 
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)             # 
    lower_bound = quartile_1 - (iqr * 1.5)  # 
    upper_bound = quartile_3 + (iqr * 1.5)  # 
    return np.where((data_out < lower_bound) | 
                    (data_out > upper_bound)) 
    

# outliers_loc1 = outliers(aaa[:,0])
# outliers_loc2 = outliers(aaa[:,1])
# print('이상치의 위치 :', outliers_loc1)
# print('이상치의 위치 :', outliers_loc2)

# outliers_loc = outliers(aaa)
# print("이상치의 위치 : ", outliers_loc)

for i in range(aaa.shape[1]):
    w = aaa[:, i]
    outliers_loc = outliers(w)
    print(i,'열의 이상치의 위치 :', outliers_loc,'\n')

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()