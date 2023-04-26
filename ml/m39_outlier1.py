import numpy as np

aaa = np.array([-10, 2,3,4,5,6,7,8,9,10,11,12,50])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quartile_1)   # 4
    print("q2 :", q2)                # 7
    print("3사분위 : ", quartile_3)   # 10
    iqr = quartile_3 - quartile_1
    print("iqr : ", iqr)             # 6
    lower_bound = quartile_1 - (iqr * 1.5)  # -5  
    upper_bound = quartile_3 + (iqr * 1.5)  # 19
    return np.where((data_out < lower_bound) | 
                    (data_out > upper_bound)) 
    
    # 19보다 크건, -5보다 작은 애 반환 = 0번째, 12번째 -10, 12

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 계산하기 쉽게하기 위해 13개를 한거야
# 양 끝을 제일 높게 해놓은거는 outlier 때문에
# 1사 분위 4
# 중위값이 7
# 3사 분위 10