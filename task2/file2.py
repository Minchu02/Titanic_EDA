import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv('Titanic-Dataset.csv')

print("First 5 rows:")
print(df.head())


print("\nDataset Info:")
print(df.info())


print("\nSummary Statistics:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())

df.hist(bins=30, figsize=(15, 10), color='skyblue')
plt.suptitle("Histograms of Numeric Features")
plt.show()


numeric_cols = df.select_dtypes(include='number').columns

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()


selected = ['Survived', 'Pclass', 'Age', 'Fare']
sns.pairplot(df[selected], hue='Survived')
plt.suptitle("Pairplot", y=1.02)
plt.show()


fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare by Survival')
fig.show()