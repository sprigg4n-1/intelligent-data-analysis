import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("banana_quality.csv", encoding="latin1")

print("2.1 Описовa статистика:")
print(df.describe())

print("3.1 Перевіряємо чи є пропущені значення:")
print(df.isnull().sum())

df_cleaned = df.dropna()
print("3.2 Перші 5 рядків після видалення пропущених значень:")
print(df_cleaned.head())

df_filled = df.fillna(df.mean(numeric_only=True))
print("4.1 Перші 5 рядків після заповнення пропущених значень:")
print(df_filled.head())

df_filled.loc[2, "Battery Capacity"] = np.nan
df_filled.loc[3, "RAM"] = np.nan
df_filled.loc[4, "Screen Size"] = np.nan

print("5.1 Додаємо пропущені значення:")
print(df_filled.head())

df_filled = df_filled.fillna(df.mean(numeric_only=True))
print("5.2 Після заповнення нових пропущених значень:")
print(df_filled.head())


print("6.1 Пошук та обробка викидів:")
print("Діаграми")

numeric_cols = ["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]

df_filled.boxplot(column=numeric_cols)
plt.show()

df_filled['Size'] = winsorize(df_filled['Size'], limits=[0.05, 0.05])
df_filled['Weight'] = winsorize(df_filled['Weight'], limits=[0.05, 0.05])
df_filled['Sweetness'] = winsorize(df_filled['Sweetness'], limits=[0.05, 0.05])
df_filled['Softness'] = winsorize(df_filled['Softness'], limits=[0.05, 0.05])
df_filled['HarvestTime'] = winsorize(df_filled['HarvestTime'], limits=[0.05, 0.05])
df_filled['Ripeness'] = winsorize(df_filled['Ripeness'], limits=[0.05, 0.05])
df_filled['Acidity'] = winsorize(df_filled['Acidity'], limits=[0.05, 0.05])

df_filled.boxplot(column=numeric_cols)
plt.show()


print("7.1 Перерахування описових:")
print(df_filled.describe())


print("8.1 Центрування та нормалізація:")
scaler = StandardScaler();

df_scaled = df_filled.copy();
df_scaled[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]] = scaler.fit_transform(df[["Size","Weight","Sweetness","Softness","HarvestTime","Ripeness","Acidity"]])

print(df_scaled)

print("8.2 Дисперсія для нормованих даних:")
print(df_scaled.var(numeric_only=True))

print("8.3 Сума для нормованих даних:")
print(df_scaled.sum(numeric_only=True))