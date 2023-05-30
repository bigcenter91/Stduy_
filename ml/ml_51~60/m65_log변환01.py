import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=1000)

#로그변환
log_data = np.log(data)

#원본 데이터 히스토그램 그리자
plt.subplot(1,2,1)
plt.hist(data, bins=50, color='blue', alpha=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.hist(log_data, bins=50, color='Red', alpha=0.5)
plt.title('Log Transformed Data')

plt.show()
# 가운데로 어떻게 모으느냐가 실력이 되는거다