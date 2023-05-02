import tensorflow as tf
import random
import numpy as np
import pandas as pd
import os
# 0. seed initialziation
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


path='d:/study/_data/AIFac_pollution'
path_list=os.listdir(path)
print(f'datafolder_list:{path_list}')

meta='/'.join([path,path_list[1]])
meta_list=os.listdir(meta)
test_aws='/'.join([path,path_list[2]])
test_aws_list=os.listdir(test_aws)
test_input='/'.join([path,path_list[3]])
test_input_list=os.listdir(test_input)
train='/'.join([path,path_list[4]])
train_list=os.listdir(train)
train_aws='/'.join([path,path_list[5]])
train_aws_list=os.listdir(train_aws)

print(f'META_list:{meta_list}')
awsmap=pd.read_csv('/'.join([meta,meta_list[0]]))
awsmap=awsmap.drop(awsmap.columns[-1],axis=1)
pmmap=pd.read_csv('/'.join([meta,meta_list[1]]))
pmmap=pmmap.drop(pmmap.columns[-1],axis=1)
print(awsmap)
print(pmmap)

#월/ 일/ 시
import pandas as pd
def split_month_day_hour(DataFrame:pd.DataFrame)->pd.DataFrame:
    month_date_time_min=[i.split(' ') for i in DataFrame['일시']]
    DataFrame=DataFrame.drop(['연도','일시'],axis=1)
    month_date=[j.split('-')for j in [i[0] for i in month_date_time_min]]
    time_min=[j.split(':')for j in[i[1] for i in month_date_time_min]]
    month=pd.Series([float(i[0]) for i in month_date],name='월')
    date=pd.Series([float(i[1]) for i in month_date],name='일')
    time=pd.Series([float(i[0])for i in time_min],name='시')
    DataFrame=pd.concat([month,date,time,DataFrame],axis=1)
    return DataFrame