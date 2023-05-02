import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pandas as pd
#1.데이터
x,y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle= True, random_state=27
)

#2. 모델
model_list = [RandomForestClassifier(),
GradientBoostingClassifier(),
DecisionTreeClassifier(),
XGBClassifier()]

# 3. 훈련
for model, value in enumerate(model_list) :
    model = value
    model.fit(x_train,y_train)

    #4. 평가, 예측
    result = model.score(x_test,y_test)
    print("acc : ", result)

    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    
    print("accuracy_score : ", acc)
    print(type(model).__name__, ":", \
        model.feature_importances_)
    print("=====================================================")
    
    # 하위 20-25%의 피처 제거 후 재학습
    #idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
    # argmin = np.argpartition(model.feature_importances_, 4)[:4]
    # x_drop = pd.DataFrame(x).drop(argmin, axis=1)
    n_drop = int(len(model.feature_importances_) * 0.25)
    idx = np.argsort(model.feature_importances_)[-n_drop:]
    x_drop = pd.DataFrame(x).drop(x.columns[idx], axis=1) 
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size=0.7, shuffle=True, random_state=123)

    model.fit(x_train1, y_train1)

    result = model.score(x_test1, y_test1)
    print("acc after feature selection: ", result)

    y_predict1 = model.predict(x_test1)
    acc1 = accuracy_score(y_test1, y_predict1)

    print("accuracy_score after feature selection: ", acc1)
    print(type(model).__name__, ":", model.feature_importances_)
    print("=====================================================")
    
# acc :  0.9740740740740741
# accuracy_score :  0.9740740740740741
# RandomForestClassifier : [0.00000000e+00 2.31677506e-03 1.81374185e-02 1.13243775e-02
#  9.15706334e-03 2.01291837e-02 9.36401592e-03 5.16144500e-04
#  4.37446615e-05 8.51152072e-03 2.60028793e-02 7.60668082e-03
#  1.37431490e-02 2.66566644e-02 5.53818973e-03 5.27583242e-04
#  3.17277445e-05 6.38150136e-03 2.40770991e-02 2.82870982e-02
#  3.37840262e-02 4.63286254e-02 8.38735231e-03 2.66509695e-04
#  2.59886601e-05 1.36257660e-02 3.96539643e-02 2.90422595e-02
#  3.57966064e-02 2.32245084e-02 2.92722438e-02 4.74122862e-05
#  0.00000000e+00 3.13111155e-02 2.74413967e-02 1.78020879e-02
#  4.13269368e-02 2.10782927e-02 2.70111178e-02 0.00000000e+00
#  0.00000000e+00 1.34324125e-02 3.69932659e-02 4.23890679e-02
#  1.73504258e-02 1.79247413e-02 2.23500136e-02 7.97623689e-05
#  0.00000000e+00 2.02166335e-03 1.64573719e-02 1.94943135e-02
#  1.25924671e-02 2.49683803e-02 2.43217877e-02 1.99008979e-03
#  0.00000000e+00 1.81588067e-03 1.96866322e-02 9.80778282e-03
#  2.73610303e-02 2.86385762e-02 1.27445134e-02 3.80079445e-03]
# =====================================================
# acc after feature selection:  0.9740740740740741
# accuracy_score after feature selection:  0.9740740740740741
# RandomForestClassifier : [3.29823014e-03 2.27710037e-02 8.17930480e-03 1.00298107e-02
#  2.04575606e-02 7.76291192e-03 5.34910301e-04 9.13244080e-05
#  9.75684764e-03 2.57010894e-02 6.93212410e-03 1.44332739e-02
#  3.11418247e-02 5.40561134e-03 7.21563160e-04 1.70565705e-04
#  7.25457413e-03 2.13267864e-02 2.43401372e-02 3.35081760e-02
#  4.43828524e-02 1.19068404e-02 3.26608795e-04 1.93767536e-04
#  1.74633118e-02 3.92259068e-02 2.32288637e-02 2.97554248e-02
#  2.35236672e-02 3.19469462e-02 7.92704655e-05 3.91872417e-02
#  2.68714568e-02 1.57855818e-02 3.42430546e-02 1.65396502e-02
#  2.51340163e-02 1.40157069e-05 9.75459080e-03 3.55806762e-02
#  4.52804814e-02 1.97324158e-02 2.09480428e-02 2.28235734e-02
#  8.51893182e-05 2.77812991e-03 1.49044703e-02 2.07837587e-02
#  1.40397198e-02 2.54593849e-02 2.76952246e-02 2.42066962e-03
#  4.39227160e-05 2.34698664e-03 1.88161311e-02 1.05130065e-02
#  2.78631022e-02 2.41556529e-02 1.76652400e-02 2.68352284e-03]
# =====================================================
# acc :  0.9611111111111111
# accuracy_score :  0.9611111111111111
# GradientBoostingClassifier : [0.00000000e+00 1.23738095e-03 1.71502982e-02 3.86564254e-03
#  1.17346889e-03 6.16242347e-02 8.16182273e-04 4.38736535e-03
#  2.43538544e-04 3.13802443e-03 1.74804320e-02 1.47537913e-03
#  6.50156894e-03 9.48560036e-03 3.12586920e-03 2.21600255e-04
#  3.15634221e-04 1.78685720e-03 1.21840157e-02 4.76050648e-02
#  2.01110596e-02 7.73758115e-02 6.01836558e-03 3.58933079e-04
#  4.28461678e-06 1.28107104e-03 4.34776688e-02 1.26936554e-02
#  3.67827741e-02 2.36925608e-02 8.90097847e-03 1.52210263e-06
#  0.00000000e+00 6.35631005e-02 5.61318248e-03 9.27028562e-03
#  7.14429164e-02 1.04887953e-02 1.91135312e-02 0.00000000e+00
#  4.60350400e-07 6.79066774e-03 7.87048128e-02 6.25404082e-02
#  1.64047242e-02 3.39888738e-02 1.57954887e-02 1.36754234e-04
#  1.46570368e-06 1.72256143e-03 5.72216200e-03 1.84867855e-02
#  9.37142760e-03 1.62734953e-02 2.75231138e-02 1.94598031e-04
#  0.00000000e+00 2.15527144e-05 6.92496733e-03 3.83417465e-03
#  6.04777366e-02 7.98164570e-03 1.64190791e-02 6.67439023e-03]
# =====================================================
# acc after feature selection:  0.9611111111111111
# accuracy_score after feature selection:  0.9611111111111111
# GradientBoostingClassifier : [7.28017550e-04 8.98083001e-03 4.81776378e-03 1.61486080e-03
#  5.62655120e-02 4.50268155e-03 2.41695598e-03 9.93852033e-04
#  3.19385080e-03 1.57598310e-02 9.11312686e-04 1.04035801e-02
#  1.22224691e-02 1.67781805e-03 3.51743125e-04 9.99675764e-05
#  3.64870131e-03 9.85329127e-03 4.07667336e-02 1.68742002e-02
#  8.15775556e-02 2.70411371e-03 2.66671102e-08 4.75282409e-05
#  5.49031960e-03 4.28764067e-02 1.63074323e-02 3.77968888e-02
#  2.37034576e-02 1.50400618e-02 7.06995939e-04 6.15775746e-02
#  7.53085531e-03 6.54309953e-03 6.50223971e-02 9.18120972e-03
#  1.34458790e-02 0.00000000e+00 4.23941148e-03 8.50829357e-02
#  7.27678425e-02 1.25229046e-02 1.64232898e-02 3.36829084e-02
#  6.56561847e-05 6.15437292e-04 1.57282557e-03 8.19448573e-03
#  1.73447762e-02 6.98328202e-03 1.18331157e-02 3.14531743e-02
#  1.25387983e-03 9.07805349e-06 1.13727534e-02 3.29507664e-03
#  5.91018436e-02 6.00598378e-03 2.61511145e-02 4.39244999e-03]
# =====================================================
# acc :  0.8555555555555555
# accuracy_score :  0.8555555555555555
# DecisionTreeClassifier : [0.         0.         0.00226181 0.         0.00295091 0.03306838
#  0.00088406 0.         0.         0.00521129 0.03491527 0.00088406
#  0.00593542 0.00173538 0.0029174  0.         0.         0.00599196
#  0.01049135 0.01695148 0.04434835 0.09661895 0.0014145  0.
#  0.00174656 0.00261311 0.07096105 0.01722152 0.05568989 0.01127215
#  0.         0.         0.         0.05336305 0.0187816  0.00380301
#  0.07606674 0.01650755 0.03126221 0.         0.         0.00646504
#  0.08154791 0.09307083 0.00088406 0.01729155 0.01004242 0.
#  0.         0.         0.01628041 0.00159131 0.00732765 0.00132609
#  0.05243165 0.0017478  0.         0.         0.00300044 0.00427295
#  0.07162407 0.         0.00434278 0.00088406]
# =====================================================
# acc after feature selection:  0.85
# accuracy_score after feature selection:  0.85
# DecisionTreeClassifier : [0.         0.01808584 0.         0.00617404 0.04757222 0.00172477
#  0.         0.         0.00088408 0.01816291 0.         0.01024407
#  0.02847549 0.         0.         0.00173399 0.00856525 0.01522754
#  0.02773282 0.05765471 0.09233409 0.         0.         0.
#  0.00250489 0.05458648 0.04044201 0.04967962 0.01011635 0.
#  0.05596559 0.0227831  0.00174659 0.07231238 0.02300827 0.00559301
#  0.         0.00696738 0.07529397 0.05966583 0.00314339 0.0022544
#  0.00783717 0.         0.         0.         0.01165328 0.
#  0.00165028 0.01471499 0.02340996 0.00282063 0.         0.
#  0.00763128 0.01216972 0.06257753 0.02838955 0.00565924 0.00085133]
# =====================================================
# acc :  0.9611111111111111
# accuracy_score :  0.9611111111111111
# XGBClassifier : [0.         0.00714994 0.0150753  0.00314903 0.0049266  0.04267235
#  0.00885519 0.04081946 0.         0.00625979 0.0132923  0.00309368
#  0.0078383  0.00848465 0.00637219 0.         0.         0.00696434
#  0.00784392 0.04127548 0.0126477  0.0453224  0.00593038 0.
#  0.         0.0071915  0.02987253 0.01014033 0.03329041 0.01864676
#  0.02231174 0.         0.         0.07378685 0.00473466 0.01014991
#  0.05983683 0.01287128 0.02885682 0.         0.         0.01283675
#  0.03499501 0.03960777 0.01636556 0.02770904 0.02593354 0.
#  0.         0.00783029 0.00399022 0.01332993 0.01484931 0.01181336
#  0.02141131 0.00180727 0.         0.00354434 0.01341767 0.00449997
#  0.0723708  0.01688889 0.03972051 0.01741586]
# =====================================================
# acc after feature selection:  0.9592592592592593
# accuracy_score after feature selection:  0.9592592592592593
# XGBClassifier : [0.         0.03586581 0.01473972 0.00630379 0.00632558 0.03811262
#  0.00687664 0.01844954 0.         0.01429335 0.01749947 0.00425431
#  0.00679329 0.00889718 0.00477541 0.00408342 0.         0.0106853
#  0.00438084 0.03571754 0.00972407 0.04290727 0.00368523 0.01006363
#  0.03107016 0.01028861 0.04547248 0.02720418 0.02153598 0.
#  0.         0.06265497 0.00255068 0.00844498 0.05418238 0.01469025
#  0.02671048 0.00813726 0.04063641 0.04245591 0.01150976 0.00681348
#  0.04784531 0.         0.         0.00532621 0.00400513 0.01504209
#  0.00431268 0.02524745 0.03434502 0.00132207 0.         0.00162754
#  0.00941294 0.00964407 0.05388613 0.01147155 0.04490386 0.01281587]
# =====================================================