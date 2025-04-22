import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.decomposition import PCA

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


# 4. Побудуйте SVM-модель та візуалізуйте її (принаймні 2 способами).
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Sales Category"] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Sales Category", palette="coolwarm")
plt.title("SVM класифікація пального через PCA")
plt.show()


# 5. Створіть прогноз на основі SVM-моделі для досліджуваних даних та оцініть його якість (як мінімум через accuracy та confusion matrix).
y_pred = svm_classifier.predict(X_test)

print("Точність:", accuracy_score(y_test, y_pred))

conf1 = confusion_matrix(y_test, y_pred)

print("Матриця плутанини:")
print(conf1)

plt.figure(figsize=(6, 5))
sns.heatmap(conf1, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Прогнозовані значення")
plt.ylabel("Реальні значення")
plt.title("Матриця плутанини SVM")
plt.show()


# 6. Підберіть оптимальні параметри моделі та оцініть її якість після перенавчання. В якості параметрів для аналізу використати всі зазначені в лекції.
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(estimator=svm.SVC(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Найкращі параметри для моделі SVM:", grid_search.best_params_)

best_svm_model = grid_search.best_estimator_
y_pred_optimized = best_svm_model.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

print(f"Точність після оптимізації: {accuracy_optimized:.2f}")

conf2 = confusion_matrix(y_test, y_pred_optimized)

print("Матриця плутанини:")
print(conf2)

# 7. Оцініть параметри "оптимальної моделі". Які з них змінились? Як саме?
print("Оптимальна модель має параметри:")
print(best_svm_model)
print("Перша модель має параметри:")
print(svm_classifier)




