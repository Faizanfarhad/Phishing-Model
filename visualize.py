import seaborn as sns
import matplotlib.pyplot as plt
import model 


df = model.df

correlation_matrix = df.corr(numeric_only=True, method='pearson')

plt.figure(figsize=(20, 5))
sns.heatmap(correlation_matrix[['label']], annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation with Label')
plt.show()