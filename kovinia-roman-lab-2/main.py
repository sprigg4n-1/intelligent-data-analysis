import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


df = pd.read_csv("car_price_dataset.csv", encoding="latin1")


print("Початкові дані")
print(df.head(20))

# 1.Здійсніть групування даних за декількома змінними, що іллюструватиме певну закономірність для досліджуваних даних. При цьому, здійсніть сортування за однією зі змінних.
print("Групування даних та сортування")
df_group = df.groupby(["Brand", "Model"]).agg({"Price": ["mean"]}).sort_values(by=("Price", 'mean'), ascending=True).reset_index()

print(df_group[["Brand", "Model", "Price"]].head())

# 2.Створіть таблицю співпряженості для будь-яких двох змінних (з врахуванням їх типу) так, щоб отримана таблиця іллюструвала певну закономірність для досліджуваних даних.
print("Таблицю співпряженості")
df_crosstab = pd.crosstab(df['Brand'], df['Transmission'])
df_crosstab_normalized = pd.crosstab(df['Brand'], df['Transmission'], normalize=True)

print("1 ---------------------")
print(df_crosstab)
print("2 ---------------------")
print(df_crosstab_normalized)

# 3. Встановити нормальність розподілу для ознак в вашому наборі даних (усіма запропонованими способами). До тих ознак, розподіл яких не є нормальним застосувати техніки приведення розподілу до нормального та візуалізувати результати до та після.
df_copy = df.copy();


rc_log = stats.boxcox(df_copy['Mileage'], lmbda=0)
rc_bc, bc_params = stats.boxcox(df_copy['Mileage'])
df_copy["rc_log1"] = rc_log
df_copy["rc_bc1"] = rc_bc

# гістограми
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig1.set_size_inches(18.5, 10.5)

df_copy["Mileage"].hist(ax=ax1, bins=100)
ax1.set_yscale('log');
ax1.tick_params(labelsize=14)
ax1.set_title('Mileage Counts Histogram', fontsize=14)
ax1.set_xlabel('')
ax1.set_ylabel('Occurrence', fontsize=14)

df_copy["rc_log1"].hist(ax=ax2, bins=100)
ax2.set_yscale('log');
ax2.tick_params(labelsize=14)
ax2.set_title('Log Transformed Counts Histogram', fontsize=14)
ax2.set_xlabel('')
ax2.set_ylabel('Occurrence', fontsize=14)

df_copy["rc_bc1"].hist(ax=ax3, bins=100)
ax3.set_yscale('log');
ax3.tick_params(labelsize=14)
ax3.set_title('Box-Cox Transformed Counts Histogram', fontsize=14)
ax3.set_xlabel('')
ax3.set_ylabel('Occurrence', fontsize=14)

# графік
print("Нормальність розподілу")
print("\t- графіки")
rc_log = stats.boxcox(df_copy['Price'], lmbda=0)
rc_bc, bc_params = stats.boxcox(df_copy['Price'])
df_copy["rc_log"] = rc_log
df_copy["rc_bc"] = rc_bc

fig2, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig2.set_size_inches(10.5, 18.5)


prob1 = stats.probplot(df_copy['Price'], dist=stats.norm, plot=ax1)
ax1.set_xlabel('');
ax1.set_title("Probplot against normal distibution")

prob2 = stats.probplot(df_copy['rc_log'], dist=stats.norm, plot=ax2)
ax2.set_xlabel('');
ax2.set_title("Probplot after log")

prob3 = stats.probplot(df_copy['rc_bc'], dist=stats.norm, plot=ax3)
ax3.set_xlabel('');
ax3.set_title("Probplot after Box-Cox")

plt.show()

# 4. Знайти кореляції (краще через візуалізацію) між усіма ознаками у датасеті та на основі цього обгрунтувати доцільність їх подальшого використання.
print("Кореляції")
print("\t- графіки")

plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.show()

# 5. Спробуйте себе у feature engineering: здійсніть перетворення одних типів даних в інші (категоріальних у числові і навпаки), здійсніть відбір значущих ознак та конструювання нових. Обгрунтуйте проведені операції 
print("Feature engineering")

df_fe = df.copy()

bins = [0, 50000, 100000, 150000, 200000, 300000]
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
df_fe['Mileage_Category'] = pd.cut(df_fe['Mileage'], bins=bins, labels=labels)
print(df_fe[['Mileage', 'Mileage_Category']].head(20))

transmission_mapping = {'Manual': 0, 'Automatic': 1, 'Semi-Automatic': 2}
df_fe['Transmission_Category'] = df_fe['Transmission'].map(transmission_mapping)
print(df_fe[['Transmission', 'Transmission_Category']].head(20))
