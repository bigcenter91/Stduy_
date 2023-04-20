import matplotlib.pyplot as plt
import seaborn as sns

print(test_data.corr())
plt.figure(figsize=(10,8))
sns.set(font_scale=1.2)
sns.heatmap(train_data.corr(), square=True, annot=True, cbar=True)
plt.show()