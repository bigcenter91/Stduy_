param_bounds = {'x1' : (-1, 5),
                'x2' : (0, 4),
                } # 중괄호 = 딕셔너리지

def y_function(x1, x2):
    return -x1 **2 - (x2 -2) **2 + 10 # 제곱이 먼저야
# 분류모델 잡아놓고 acc 하면되겠지
# 회귀일 땐 r2를 보조로하고 rmse, mae, mse를 쓰고 -를 하면 되겠지

# 이 함수의 최대값을 찾을거다 x1은 0, x2는 2가 되야겠지

# pip install Bayesian-Optimization
from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds=param_bounds,
    random_state=337
)

# 'x1' 텍스트 형태로 넣어주고
# (-1, 5) 튜플 형태로 넣어줘야한다

optimizer.maximize(init_points=5,
                   n_iter=100) #2개의 값을 20번 찾을거야

print(optimizer.max) # 분홍색 글씨가 갱신된걸 뜻하는거야
# -------------------------------------------------
# | 1         | -6.356    | 3.799     | 0.6143    |
# | 2         | 0.9635    | 2.407     | 0.1988    |
# | 3         | -3.961    | 3.207     | 0.08218   |
# | 4         | 0.6734    | 2.498     | 0.2429    |
# | 5         | 6.848     | 1.504     | 1.057     |
# | 6         | 9.805     | 0.2989    | 2.325     |
# | 7         | 3.928     | 1.439     | 4.0       |
# | 8         | 7.704     | -1.0      | 0.8618    |
# | 9         | 5.281     | -1.0      | 3.928     |
# | 10        | 8.955     | -1.0      | 2.211     |
# | 11        | 9.723     | 0.04416   | 1.475     |
# | 12        | 9.937     | -0.2417   | 2.065     |
# | 13        | 9.758     | 0.4758    | 1.874     |
# | 14        | 9.993     | 0.07269   | 1.957     |
# | 15        | 9.811     | -0.1515   | 2.407     |
# | 16        | 9.985     | 0.01247   | 2.122     |
# | 17        | 9.985     | -0.07275  | 1.899     |
# | 18        | 9.999     | -0.02136  | 2.009     |
# | 19        | 10.0      | -0.01201  | 2.005     |
# | 20        | 10.0      | -0.006671 | 2.002     |
# | 21        | 10.0      | 0.0006546 | 2.0       |
# | 22        | 9.998     | 0.04309   | 2.008     |
# =================================================