import h2o

h2o.init()

train = h2o.import_file("c:/study_data/_data/dacon_book/train.csv")
test = h2o.import_file("c:/study_data/_data/dacon_book/test.csv")

x = train.columns
y = "Book-Rating"
x.remove(y)

from h2o.automl import H2OAutoML

aml = H2OAutoML(
    max_models=10,
    seed=42,
    max_runtime_secs=360,
    sort_metric='RMSE'

)

aml.train(
    x=x,
    y=y,
    training_frame=train
)

leaderboard = aml.leaderboard
print(leaderboard.head())


test = h2o.import_file("c:/study_data/_data/dacon_book/content/test.csv")

len(test)
model = aml.leader

pred = model.predict(test)

pred_df = pd.DataFrame(pred.as_data_frame())
pred_df['predict']

import pandas as pd
submit = pd.read_csv('c:/study_data/_data/dacon_book/sample_submission.csv')

