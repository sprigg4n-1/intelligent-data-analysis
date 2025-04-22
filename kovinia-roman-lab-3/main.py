import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# 1. Завантажте власні дані. З врахуванням вимог, що ставляться до даних алгоритмом і відповідно до поставленої задачі досліджень.
df = pd.read_csv("../DatasetForCoffeeSales2.csv", encoding="latin1")


label_encoder = LabelEncoder()
scaler = StandardScaler()

df_copy = df.copy()
numeric_cols = ['Unit Price','Quantity','Sales Amount','Discount_Amount','Final Sales']

print("Початкові дані")
print(df_copy.describe(include='all'))
print(df_copy.head(20))


# 2. Підготуйте дані з використанням підходів з попередніх лабораторних робіт.
# перевірка на нульові
print(df_copy.isnull().sum())

if (df_copy.isnull().sum().sum() > 0):
    df_copy = df.fillna(df.mean(numeric_only=True))
    df_copy = df.fillna("Unkown", numeric_cols=False)

# перевірка на викиди та їх усунення
df_copy.boxplot(column=numeric_cols)
plt.show()

for i in numeric_cols:
    df_copy[i] = winsorize(df_copy[i], limits=[0.05, 0.05])
    print(i)

df_copy.boxplot(column=numeric_cols)
plt.show()

# центрування та нормалізація
scaler = StandardScaler()

df_copy[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# кореляція і feature engineering
df_copy["Category"] = label_encoder.fit_transform(df_copy["Category"])
df_copy["Product"] = label_encoder.fit_transform(df_copy["Product"])
df_copy["City"] = label_encoder.fit_transform(df_copy["City"])

threshold = df_copy['Final Sales'].median()
df_copy['Sales Category'] = ['High' if x > threshold else 'Low' for x in df_copy['Final Sales']]

# 3. Розбийте ваш набір даних на навчальну та тестові підгрупи з використанням всіх відомих вам підходів.

X = df_copy.drop(columns=['Final Sales', 'Date', 'Customer_ID', 'Sales Category', 'Discount_Amount', 'Used_Discount'])
y = df_copy["Sales Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# 4. Отримайте дерева рішень, що міститиме набір вирішуючих правил для класифікації та прогнозу даних.
dt_model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=100, min_samples_leaf=2)
dt_model_gini.fit(X_train, y_train)

# Візуалізація дерева рішень
plt.figure(figsize=(14, 7))
tree.plot_tree(dt_model_gini, feature_names=X_train.columns, filled=True, fontsize=7)
plt.show()

# 5. Оцініть якість отриманої моделі (принаймні 2-ма способами).
y_pred = dt_model_gini.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точність моделі: {accuracy:.2f}')

cm = confusion_matrix(y_test, y_pred)
print("Матриця плутанини:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Матриця плутанини")
plt.xlabel("Передбачив")
plt.ylabel("Правда")
plt.show()

# 6. Підберіть оптимальні параметри моделі, зокрема, оптимальну складність.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

min_samples_leaf = range(1, 10, 1)
max_features = ['sqrt', 'log2'] 
min_samples_split = range(2, 10, 2)
max_depth = range(2, 20, 1)

param_grid = dict(
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,
    max_features=max_features
)

dt_model_optimal = DecisionTreeClassifier(random_state=100)

kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)

grid_search = GridSearchCV(estimator=dt_model_optimal, param_grid=param_grid, scoring="accuracy", cv=kfold, n_jobs=-1, verbose=1)

grid_result = grid_search.fit(X, y_encoded)
print("Найкращі: %f використовуються %s" % (grid_result.best_score_, grid_result.best_params_))

# 7. Виконайте обрізку та перебудову дерева, відповідно до параметрів, визначених у пункті 6.
best_params = grid_result.best_params_
best_model = DecisionTreeClassifier(
    max_depth=best_params['max_depth'], 
    max_features=best_params['max_features'], 
    min_samples_leaf=best_params['min_samples_leaf'], 
    min_samples_split=best_params['min_samples_split'],
    random_state=100
)
best_model.fit(X_train, y_train)

plt.figure(figsize=(14, 7))
tree.plot_tree(best_model, feature_names=X_train.columns, filled=True, fontsize=7)
plt.show()

# 8. Оцініть параметри "оптимальної моделі". Які з них змінились? Як саме?
init_params = dt_model_gini.get_params()
optimal_params = best_model.get_params()

for param in init_params:
    if init_params[param] != optimal_params.get(param, None):
        print(f"Змінено параметр: {param}")
        print(f"Початкове значення: {init_params[param]}")
        print(f"Оптимальне значення: {optimal_params.get(param, 'не задано')}\n")

# 9. Встановіть важливість ознак для кінцевого результату. Здійсніть їх відбір. Проведіть перенавчання на оцінку якості роботи моделі. Порівняйте з раніше отриманими результатами.
feature_importances = best_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print('Важливість')
print(importance_df)

threshold = 0.05
important_features = importance_df[importance_df['Importance'] > threshold]['Feature'].tolist()
print('Відбір')
print(important_features)

X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Permutation Importance')
plt.show()

# Створимо нову модель з оптимальними параметрами
best_model_selected = DecisionTreeClassifier(
    max_depth=best_params['max_depth'], 
    max_features=best_params['max_features'], 
    min_samples_leaf=best_params['min_samples_leaf'], 
    min_samples_split=best_params['min_samples_split'],
    random_state=100
)

best_model_selected.fit(X_train_selected, y_train)

y_pred_selected = best_model_selected.predict(X_test_selected)

cm = confusion_matrix(y_test, y_pred_selected)
print("Матриця плутанини:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Матриця плутанини")
plt.xlabel("Передбачив")
plt.ylabel("Правда")
plt.show()

accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f'Точність на тестовій вибірці після перенавчання: {accuracy_selected:.2f}')
print(f'Точність до відбору ознак: {accuracy:.2f}')
