import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

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

# 3. Розбийте ваш набір даних на навчальну та тестові підгрупи з використанням всіх відомих вам підходів.
threshold = df_copy['Final Sales'].median()
df_copy['Sales Category'] = ['High' if x > threshold else 'Low' for x in df_copy['Final Sales']]

X = df_copy.drop(columns=['Final Sales', 'Date', 'Customer_ID', 'Sales Category'])
y = df_copy["Sales Category"]
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 4. Створіть класифікаційну та прогнозну моделі для досліджуваних даних на основі k-NN алгоритму. 
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Класифікаційна модель k-NN
knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(X_train, y_train)

y_pred_class = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Точність класифікації за допомогою k-NN: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Прогнозовані значення")
plt.ylabel("Реальні значення")
plt.title("Матриця плутанини для моделі k-NN")
plt.show()

# Візуалізація класифікації
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train) 

knn_classifier = KNeighborsClassifier(n_neighbors=2)
knn_classifier.fit(X_train_pca, y_train_encoded) 

plt.figure(figsize=(8, 6))
plot_decision_regions(X=X_train_pca, y=y_train_encoded, clf=knn_classifier, legend=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KNN Classification (K=5)')

plt.show()

# Регресійна модель k-NN
knn_regressor = KNeighborsRegressor(n_neighbors=15)
knn_regressor.fit(X_train, y_train_encoded) 

y_pred_r = knn_regressor.predict(X_test)
mse = mean_squared_error(y_test_encoded, y_pred_r)
print(f"MSE регресійної моделі: {mse:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_encoded, y_pred_r, color='blue', label='Передбачені vs Реальні', alpha=0.6)
plt.plot([min(y_test_encoded), max(y_test_encoded)], [min(y_test_encoded), max(y_test_encoded)], color='red', linestyle='--', label='Ідеальна лінія')
plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.title('Порівняння реальних та передбачених значень для регресійної моделі k-NN')
plt.legend()
plt.show()

# 5. Підіберіть оптимальні параметри моделі для досліджуваних даних.
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],  # Список можливих значень k
    'weights': ['uniform', 'distance'],  # Ваги для сусідів
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Метрики відстані
}

grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Найкращі параметри для класифікаційної моделі k-NN:", grid_search.best_params_)

best_knn_classifier = grid_search.best_estimator_
y_pred_class_optimized = best_knn_classifier.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_class_optimized)
print(f"Оптимізована точність класифікації: {accuracy_optimized:.2f}")

grid_search_regressor = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

# Навчання моделі
grid_search_regressor.fit(X_train, y_train_encoded)

# Виведення найкращих параметрів
print("Найкращі параметри для регресійної моделі k-NN:", grid_search_regressor.best_params_)

# Оцінка MSE з найкращими параметрами
best_knn_regressor = grid_search_regressor.best_estimator_
y_pred_r_optimized = best_knn_regressor.predict(X_test)
mse_optimized = mean_squared_error(y_test_encoded, y_pred_r_optimized)
print(f"Оптимізований MSE для регресійної моделі: {mse_optimized:.4f}")

# 6. Візуалізуйте початкову та "оптимальну" моделі.
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

best_knn_classifier.fit(X_train_pca, y_train_encoded)

plt.figure(figsize=(8, 6))
plot_decision_regions(X=X_train_pca, y=y_train_encoded, clf=best_knn_classifier, legend=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Оптимізована модель KNN (K={grid_search.best_params_["n_neighbors"]})')
plt.show()

# регресія
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_class, color='blue', label='Передбачені vs Реальні (початково)', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ідеальна лінія')

plt.xlabel('Реальні значення')
plt.ylabel('Передбачені значення')
plt.title('Порівняння реальних та передбачених значень для початкової класифікаційної моделі k-NN')
plt.legend()
plt.show()
