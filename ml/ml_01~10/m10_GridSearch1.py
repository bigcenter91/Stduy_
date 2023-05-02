import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# 아이리스 데이터하고 트레인테스트스플릿하고 KFold, cross_val_score 전처리하는구나를 알아야한다

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, stratify=y
)
# 30개가 테스트가 될거다

gamma = [0.001, 0.01, 0.1, 1, 10, 100]
C = [0.001, 0.01, 0.1, 1, 10, 100]
# C 곡선 값에 다가가서 그린다?

max_score = 0
for i in gamma: #감마만 6번 돌겠지?
    for j in C:
        
        #2. 모델
        model = SVC(gamma=i, C=j)
        #gamma=gamma, C=C 이렇게 할 수 없으니 for문 돌려야한다

        #3. 컴파일, 훈련
        model.fit(x_train, y_train)

        #4. 평가, 예측
        # Evaluate = score
        score = model.score(x_test, y_test)
              
        if max_score < score :
            max_score = score
            best_parameters = {'gamma': i, 'C' :j } # 이때 gamma와 C는 항상 같이 간다 왜? if문 안에 있으니까


print("최고점수 : ", max_score)
print("최적의 매개변수 :", best_parameters)
#acc :  0.9666666666666667

