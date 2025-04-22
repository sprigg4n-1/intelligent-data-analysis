import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Завантажте власні дані. З врахуванням вимог, що ставляться до даних алгоритмом і відповідно до поставленої задачі досліджень.
df = pd.read_csv("../DatasetForCoffeeSales2.csv", encoding="latin1")

numeric_cols = ['Unit Price','Quantity','Sales Amount','Discount_Amount','Final Sales']

label_encoder = LabelEncoder()
scaler = StandardScaler()

df_copy = df.copy();
# 2. Підготуйте дані з використанням підходів з попередніх лабораторних робіт (2.1 та 2.2).
print(df_copy.isnull().sum())

df_copy = df.fillna(df.mean(numeric_only=True))
df_copy = df_copy.fillna("Unknown") 

df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])

df_copy["Category"] = label_encoder.fit_transform(df_copy["Category"])
df_copy["Product"] = label_encoder.fit_transform(df_copy["Product"])
df_copy["City"] = label_encoder.fit_transform(df_copy["City"])

threshold = df_copy['Final Sales'].median()
df_copy['Sales Category'] = ['High' if x > threshold else 'Low' for x in df_copy['Final Sales']]

# 3. Розбийте ваш набір даних на навчальну та тестові підгрупи з використанням всіх відомих вам підходів.
X_one = df_copy[['Sales Amount']]
X_multiple = df_copy.drop(columns=['Final Sales', 'Date', 'Customer_ID', 'Sales Category', 'Sales Amount'])

y = df_copy['Final Sales']

print('-------------------------------- Створіть прогнозну моделі для досліджуваних даних на основі лінійної регресії')
# 4.Створіть прогнозну моделі для досліджуваних даних на основі лінійної регресії:
# - з використанням одного Х;
X_train_one, X_test_one, y_train, y_test = train_test_split(X_one, y, test_size=0.3, random_state=100)


X_train_one_scaled = scaler.fit_transform(X_train_one)
X_test_one_scaled = scaler.transform(X_test_one)

lin_reg_one = LinearRegression()
lin_reg_one.fit(X_train_one_scaled, y_train)

y_pred_one = lin_reg_one.predict(X_test_one_scaled)

mse_one = mean_squared_error(y_test, y_pred_one)
r2_one = r2_score(y_test, y_pred_one)

print(f"МSE для лінійної регресії з одним X (Quantity): {mse_one:.4f}")

# - з використанням кількох Х-ів
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(X_multiple, y, test_size=0.3, random_state=100)

X_train_multiple_scaled = scaler.fit_transform(X_train_multiple)
X_test_multiple_scaled = scaler.transform(X_test_multiple)

lin_reg_multiple = LinearRegression()
lin_reg_multiple.fit(X_train_multiple_scaled, y_train)

y_pred_multiple = lin_reg_multiple.predict(X_test_multiple_scaled)

mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print(f"МSE для лінійної регресії з кількома X: {mse_multiple:.4f}")


# 5. Підіберіть оптимальні параметри створених моделей для досліджуваних даних (параметри регуляризації, гребенева та лассо регресія)
print('-------------------------------- Підіберіть оптимальні параметри створених моделей для досліджуваних даних ONE_X')
ridge_one = Ridge()
ridge_param_grid_one = {'alpha': np.logspace(-6, 6, 13)}  # Шукаємо параметри alpha від 10^-6 до 10^6
ridge_grid_search_one = GridSearchCV(ridge_one, ridge_param_grid_one, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid_search_one.fit(X_train_one_scaled, y_train)

# Для лассо регресії (Lasso) з одним X
lasso_one = Lasso()
lasso_param_grid_one = {'alpha': np.logspace(-6, 6, 13)}  # Шукаємо параметри alpha від 10^-6 до 10^6
lasso_grid_search_one = GridSearchCV(lasso_one, lasso_param_grid_one, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid_search_one.fit(X_train_one_scaled, y_train)

# Вивести результати для моделі з одним X
print(f"Оптимальний alpha для Ridge з одним X: {ridge_grid_search_one.best_params_['alpha']}")
print(f"Найкраще MSE для Ridge з одним X: {-ridge_grid_search_one.best_score_:.4f}")

print(f"Оптимальний alpha для Lasso з одним X: {lasso_grid_search_one.best_params_['alpha']}")
print(f"Найкраще MSE для Lasso з одним X: {-lasso_grid_search_one.best_score_:.4f}")

# Для гребеневої регресії (Ridge) з кількома X
print('-------------------------------- Підіберіть оптимальні параметри створених моделей для досліджуваних даних MULTIPLY_X')
ridge_multiple = Ridge()
ridge_param_grid_multiple = {'alpha': np.logspace(-6, 6, 13)}  # Шукаємо параметри alpha від 10^-6 до 10^6
ridge_grid_search_multiple = GridSearchCV(ridge_multiple, ridge_param_grid_multiple, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid_search_multiple.fit(X_train_multiple_scaled, y_train)

# Для лассо регресії (Lasso) з кількома X
lasso_multiple = Lasso()
lasso_param_grid_multiple = {'alpha': np.logspace(-6, 6, 13)}  # Шукаємо параметри alpha від 10^-6 до 10^6
lasso_grid_search_multiple = GridSearchCV(lasso_multiple, lasso_param_grid_multiple, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid_search_multiple.fit(X_train_multiple_scaled, y_train)

# Вивести результати для моделі з кількома X
print(f"Оптимальний alpha для Ridge з кількома X: {ridge_grid_search_multiple.best_params_['alpha']}")
print(f"Найкраще MSE для Ridge з кількома X: {-ridge_grid_search_multiple.best_score_:.4f}")

print(f"Оптимальний alpha для Lasso з кількома X: {lasso_grid_search_multiple.best_params_['alpha']}")
print(f"Найкраще MSE для Lasso з кількома X: {-lasso_grid_search_multiple.best_score_:.4f}")

# 6. Візуалізуйте початкові та "оптимальні" моделі.
# Візуалізація лінійної регресії з одним X
plt.figure(figsize=(14, 6))

# Лінійна регресія з одним X (Quantity)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_one, color='blue', label='Прогнозовані значення')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title("Лінійна регресія з одним X (Quantity)")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

# Лінійна регресія з кількома X
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_multiple, color='green', label='Прогнозовані значення')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title("Лінійна регресія з кількома X")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

plt.tight_layout()
plt.show()

# Візуалізація оптимальних моделей (Ridge та Lasso для одного X)
plt.figure(figsize=(14, 6))

# Ridge для одного X
y_pred_ridge_one = ridge_grid_search_one.best_estimator_.predict(X_test_one_scaled)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge_one, color='orange', label='Прогнозовані значення Ridge')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title(f"Ridge регресія з одним X (α={ridge_grid_search_one.best_params_['alpha']})")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

# Lasso для одного X
y_pred_lasso_one = lasso_grid_search_one.best_estimator_.predict(X_test_one_scaled)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso_one, color='purple', label='Прогнозовані значення Lasso')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title(f"Lasso регресія з одним X (α={lasso_grid_search_one.best_params_['alpha']})")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

plt.tight_layout()
plt.show()

# Візуалізація оптимальних моделей (Ridge та Lasso для кількох X)
plt.figure(figsize=(14, 6))

# Ridge для кількох X
y_pred_ridge_multiple = ridge_grid_search_multiple.best_estimator_.predict(X_test_multiple_scaled)
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ridge_multiple, color='orange', label='Прогнозовані значення Ridge')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title(f"Ridge регресія з кількома X (α={ridge_grid_search_multiple.best_params_['alpha']})")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

# Lasso для кількох X
y_pred_lasso_multiple = lasso_grid_search_multiple.best_estimator_.predict(X_test_multiple_scaled)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso_multiple, color='purple', label='Прогнозовані значення Lasso')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Лінія ідеальної відповідності')
plt.title(f"Lasso регресія з кількома X (α={lasso_grid_search_multiple.best_params_['alpha']})")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.legend()

plt.tight_layout()
plt.show()


print('-------------------------------- Коефіцієнти')
# Лінійна регресія з одним X (Quantity)
print(f"Лінійна регресія з одним X (Quantity):")
print(f"  Перехід (a): {lin_reg_one.intercept_:.4f}")
print(f"  Коефіцієнт (b): {lin_reg_one.coef_[0]:.4f}")

# Лінійна регресія з кількома X
print(f"Лінійна регресія з кількома X:")
print(f"  Перехід (a): {lin_reg_multiple.intercept_:.4f}")
print(f"  Коефіцієнти (b): {', '.join([f'{coef:.4f}' for coef in lin_reg_multiple.coef_])}")

# Ridge для одного X
print(f"Ridge регресія з одним X (α={ridge_grid_search_one.best_params_['alpha']}):")
print(f"  Перехід (a): {ridge_grid_search_one.best_estimator_.intercept_:.4f}")
print(f"  Коефіцієнт (b): {ridge_grid_search_one.best_estimator_.coef_[0]:.4f}")

# Lasso для одного X
print(f"Lasso регресія з одним X (α={lasso_grid_search_one.best_params_['alpha']}):")
print(f"  Перехід (a): {lasso_grid_search_one.best_estimator_.intercept_:.4f}")
print(f"  Коефіцієнт (b): {lasso_grid_search_one.best_estimator_.coef_[0]:.4f}")

# Ridge для кількох X
print(f"Ridge регресія з кількома X (α={ridge_grid_search_multiple.best_params_['alpha']}):")
print(f"  Перехід (a): {ridge_grid_search_multiple.best_estimator_.intercept_:.4f}")
print(f"  Коефіцієнти (b): {', '.join([f'{coef:.4f}' for coef in ridge_grid_search_multiple.best_estimator_.coef_])}")

# Lasso для кількох X
print(f"Lasso регресія з кількома X (α={lasso_grid_search_multiple.best_params_['alpha']}):")
print(f"  Перехід (a): {lasso_grid_search_multiple.best_estimator_.intercept_:.4f}")
print(f"  Коефіцієнти (b): {', '.join([f'{coef:.4f}' for coef in lasso_grid_search_multiple.best_estimator_.coef_])}")
