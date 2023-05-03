# pip install hyperopt
# hyper는 최소값
# 베이지안은 최대값을 찾는거고
import hyperopt 
print(hyperopt.__version__) # 0.2.7
import numpy as np 
from hyperopt import hp, fmin, Trials, tpe, STATUS_OK
import pandas as pd


search_space = {
    'x1' : hp.quniform('x', -10, 10, 1),
    'x2' : hp.quniform('x2', -15, 15, 1) # 정규분포일 때 유니폼이야
    #      hp.quniform(label, low, high, q)
}
print(search_space)

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 -20*x2

    return return_value
    # 권장리턴방식 return {'loss' : return_value, 'status':STATUS_OK}
trial_val = Trials()

best = fmin(
    fn=objective_func,
    space=search_space,
    algo = tpe.suggest,
    max_evals=10,
    trials = trial_val,
    rstate = np.random.default_rng(seed=10),
)
    
print('best : ', best)
# 
print(trial_val.results)
print(trial_val.vals)



######### pandas 데이터프레임에 trial_val.vals를 넣어라!!! #########

a = pd.DataFrame(trial_val.vals)
print(a)

results = [aaa['loss'] for aaa in trial_val.results]

# for loss_dict in trial_val.results
df = pd.DataFrame({'x1': trial_val.vals['x1'],
                   'x2': trial_val.vals['x2'],
                   'results': results})
print(df)