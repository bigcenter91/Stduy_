import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

path = "d:/study_data/_data/project_p/"
save_path = "d:/study_data/_save/project_p/"

# Load weather data from entire August months for both years
data_region1 = pd.concat([
    pd.read_csv(path + "2108_광주날씨.csv", index_col=0, encoding='cp949'),
    pd.read_csv(path +"2208_광주날씨.csv", index_col=0, encoding='cp949')
])
data_region2 = pd.concat([
    pd.read_csv(path +"2108_전주날씨.csv", index_col=0, encoding='cp949'),
    pd.read_csv(path +"2208_전주날씨.csv", index_col=0, encoding='cp949')
])
data_region3 = pd.concat([
    pd.read_csv(path +"2108_목포날씨.csv", index_col=0, encoding='cp949'),
    pd.read_csv(path +"2208_목포날씨.csv", index_col=0, encoding='cp949')
])

# Extract the relevant features and target for each region
x_region1 = data_region1[["평균기온(°C)", "평균 상대습도(%)", "평균 풍속(m/s)"]]
y_region1 = data_region1["일강수량(mm)"]

x_region2 = data_region2[["평균기온(°C)", "평균 상대습도(%)", "평균 풍속(m/s)"]]
y_region2 = data_region2["일강수량(mm)"]

x_region3 = data_region3[["평균기온(°C)", "평균 상대습도(%)", "평균 풍속(m/s)"]]
y_region3 = data_region3["일강수량(mm)"]

# Train a linear regression model for each region
reg_region1 = LinearRegression().fit(x_region1, y_region1)
reg_region2 = LinearRegression().fit(x_region2, y_region2)
reg_region3 = LinearRegression().fit(x_region3, y_region3)

# Make a prediction for August 2023 for each region
prediction_region1 = reg_region1.predict([[25, 70, 10]] * 31)  # Assuming 31 days in August
prediction_region2 = reg_region2.predict([[25, 70, 10]] * 31)  # Assuming 31 days in August
prediction_region3 = reg_region3.predict([[25, 70, 10]] * 31)  # Assuming 31 days in August

print("Predicted rainfall in August 2023 for Region 1:", prediction_region1.sum())
print("Predicted rainfall in August 2023 for Region 2:", prediction_region2.sum())
print("Predicted rainfall in August 2023 for Region 3:", prediction_region3.sum())