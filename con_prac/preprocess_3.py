import os
import numpy as np
import pandas as pd
from haversine import haversine
from typing import Tuple

def bring(filepath:str)->pd.DataFrame:
    li = []
    for i in filepath:
        df = pd.read_csv(i, index_col=None, header=0, encoding='utf-8-sig')
        li.append(df)
    data = pd.concat(li, axis=0, ignore_index=True)
    return data

def load_aws_and_pm()->Tuple[pd.DataFrame, pd.DataFrame]:
    path='c:/study_data/_data/finedust/'
    path_list = os.listdir(path)

    meta='/'.join([path, path_list[1]])
    meta_list=os.listdir(meta)

    awsmap = pd.read_csv('/'.join([meta,meta_list[0]]))
    awsmap = awsmap.drop(awsmap.columns[-1], axis=1)
    pmmap = pd.read_csv('/'.join([meta,meta_list[1]]))
    pmmap = pmmap.drop(pmmap.columns[-1], axis=1)
    return awsmap, pmmap

def distance(awsmap:pd.DataFrame,pmmap:pd.DataFrame)->pd.DataFrame:
    '''pm과 ams관측소 사이의 거리들을 프린트해준다'''
    a = []
    for i in range(pmmap.shape[0]):
        b=[]
        for j in range(awsmap.shape[0]):
            b.append(haversine((np.array(pmmap)[i, 1], np.array(pmmap)[i, 2]), (np.array(awsmap)[j, 1], np.array(awsmap)[j, 2])))
        a.append(b)
    distance = pd.DataFrame(np.array(a),index=pmmap['Location'],columns=awsmap['Location'])
    return distance

def scaled_score(distance:pd.DataFrame,pmmap:pd.DataFrame,near:int=3)->Tuple[pd.DataFrame,np.ndarray]:
    '''pm으로부터 가까운 상위 near개의 환산점수'''
    min_i=[]
    min_v=[]
    for i in range(distance.shape[0]):
        min_i.append(np.argsort(distance.values[i,:])[:near])
        min_v.append(distance.values[i, min_i[i]])

    min_i = np.array(min_i)
    min_v = pd.DataFrame(np.array(min_v),index=distance.index)
    
    for i in range(pmmap.shape[0]):
        for j in range(near):
            min_v.values[i, j]=min_v.values[i, j]**2
            
    sum_min_v = np.sum(min_v, axis=1)

    recip=[]
    for i in range(pmmap.shape[0]):
        recip.append(sum_min_v[i]/min_v.values[i, :])
    recip = np.array(recip)
    recip_sum = np.sum(recip, axis=1)
    coef = 1/recip_sum

    result = []
    for i in range(pmmap.shape[0]):
        result.append(recip[i, :]*coef[i])
    result = pd.DataFrame(np.array(result),index=distance.index)
    return result, min_i

def split_x(dt, ts, pred_date):
    a = []
    for j in range(dt.shape[0]):
        b = []
        for i in range(dt.shape[1]-ts-pred_date):
            c = dt[j, i:i+ts, :]
            b.append(c)
        a.append(b)
    return np.array(a)

def get_hourly_features(hour: int):
    """주어진 시간(hour)에 대한 사인과 코사인 함수 값을 반환"""
    radians_per_hour = 2 * np.pi * hour / 24.0
    return [np.sin(radians_per_hour), np.cos(radians_per_hour)]