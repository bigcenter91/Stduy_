import pandas as pd
from sklearn.ensemble import IsolationForest

# Load train and test data
path= "d:/study_data/_data/fac/"
save_path=  "d:/study_data/_save/fac/"
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

train_data = train_data.drop(['out_pressure'],axis=1)
test_data = test_data.drop(['out_pressure'],axis=1)
# Combine train and test data
# data = pd.concat([train_data, test_data], axis=0)

# Preprocess data
# ...
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])






# Train isolation forest model on train data
model = IsolationForest(n_estimators=3000,random_state=333,max_samples=2000,
                        max_features=7, bootstrap=False,)

model.fit(train_data)

# Predict anomalies in test data
predictions = model.predict(test_data)

# Save predictions to submission file
new_predictions = [0 if x == 1 else 1 for x in predictions]
submission['label'] = pd.DataFrame({'Prediction': new_predictions})
submission.to_csv(save_path+'submit0404_3.csv', index=False)