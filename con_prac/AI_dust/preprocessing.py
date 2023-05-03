import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor


def split_month_day_hour(DataFrame:pd.DataFrame)->pd.DataFrame:
    month_date_time_min=[i.split(' ') for i in DataFrame['일시']]
    DataFrame=DataFrame.drop(['연도','일시'],axis=1)
    month_date=[j.split('-')for j in [i[0] for i in month_date_time_min]]
    time_min=[j.split(':')for j in[i[1] for i in month_date_time_min]]
    month=pd.Series([float(i[0]) for i in month_date],name='월', index=DataFrame.index)
    date=pd.Series([float(i[1]) for i in month_date],name='일', index=DataFrame.index)
    time=pd.Series([float(i[0])for i in time_min],name='시', index=DataFrame.index)
    DataFrame=pd.concat([month,date,time,DataFrame],axis=1)
    return DataFrame


def Imputation(DataFrame:pd.DataFrame)->pd.DataFrame:
    imputer=IterativeImputer(XGBRegressor(tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0,n_estimators=1000,learning_rate=0.001,subsample=1,
                    max_depth=8))
    DataFrame=pd.DataFrame(imputer.fit_transform(DataFrame),columns=DataFrame.columns)
    DataFrame=DataFrame.interpolate()
    DataFrame=DataFrame.fillna(method='ffill')
    DataFrame=DataFrame.fillna(method='bfill')
    return DataFrame

# n_estimators=600,learning_rate=0.01,subsample=1, max_depth=7))  // 9.87202882
# n_estimators=600,learning_rate=0.001,subsample=1, max_depth=7)) // 62.66545394	
# n_estimators=1000,learning_rate=0.001,subsample=1, max_depth=7)) // 37.28416054	
# n_estimators=700,learning_rate=0.01,subsample=1, max_depth=8)) // 9.78048726