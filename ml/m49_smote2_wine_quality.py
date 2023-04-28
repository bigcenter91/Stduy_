# [실습] smote 적용

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine # 증폭시킬려면 쪼개서 삭제를 해야하기 때문에
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE     
     
     
#1 데이터     
datasets = load_wine()
x = datasets.data
y = datasets['target']
