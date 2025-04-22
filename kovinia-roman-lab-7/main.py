import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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

def categorize_quantity(x):
    if x <= -1:
        return 'Very Low'
    elif x <= 0:
        return 'Low'
    elif x <= 1:
        return 'Medium'
    else:
        return 'High'


df_copy['Quantity Category'] = df_copy['Quantity'].apply(categorize_quantity)
df_copy['Quantity Category'] = label_encoder.fit_transform(df_copy['Quantity Category'])


# 3. Розбийте ваш набір даних на навчальну та тестові підгрупи з використанням всіх відомих вам підходів.
X = df_copy.drop(columns=['Final Sales', 'Date', 'Customer_ID', 'Sales Category', 'Quantity Category'])
y_binary = df_copy['Sales Category']
y_multiclass = df_copy['Quantity Category']

# 4. Побудуйте модель логістичної регресії
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

# - для бінарної класифікацї;
log_reg_bin = LogisticRegression()
log_reg_bin.fit(X_train, y_train_bin)
y_pred_bin = log_reg_bin.predict(X_test)

# - для множинної класифікаціїї
log_reg_multi = LogisticRegression(solver='saga', max_iter=500)
log_reg_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = log_reg_multi.predict(X_test_multi)


# 5. Оцініть якість отриманої моделі (принаймні 2-ма способами). 
print("Точність (Бінарна класифікація):", accuracy_score(y_test_bin, y_pred_bin))

conf_matrix = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогнозований")
plt.ylabel("Насправді")
plt.title("Матриця плутанини - Бінарна")
plt.show()

print("Точність (Множинна класифікація):", accuracy_score(y_test_multi, y_pred_multi))

conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(conf_matrix_multi, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогнозований")
plt.ylabel("Насправді")
plt.title("Матриця плутанини - Множинна")
plt.show()

# 6. Підберіть оптимальні параметри моделей ('penalty', С, 'solver')
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs']
}

grid_search_bin = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search_bin.fit(X_train, y_train_bin)

grid_search_multi = GridSearchCV(LogisticRegression(solver='saga'), param_grid, cv=5, scoring='accuracy')
grid_search_multi.fit(X_train_multi, y_train_multi)

# 7. Перенавчіть моделі відповідно до оптимальних параметрів, визначених у пункті 6.
log_reg_bin = LogisticRegression(**grid_search_bin.best_params_)
log_reg_bin.fit(X_train, y_train_bin)
y_pred_bin = log_reg_bin.predict(X_test)

conf_matrix_bin = confusion_matrix(y_test_bin, y_pred_bin)
sns.heatmap(conf_matrix_bin, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогнозований")
plt.ylabel("Насправді")
plt.title("Матриця плутанини - Бінарна")
plt.show()

log_reg_multi = LogisticRegression(**grid_search_multi.best_params_)
log_reg_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = log_reg_multi.predict(X_test_multi)

conf_matrix_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(conf_matrix_multi, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогнозований")
plt.ylabel("Насправді")
plt.title("Матриця плутанини - Множинна")
plt.show()

print("Найкращі параметри для бінарної:", grid_search_bin.best_params_)
print("Найкращі параметри для множинної:", grid_search_multi.best_params_)
print("Точність оптимальної (бінарної):", accuracy_score(y_test_bin, y_pred_bin))
print("Точність оптимальної (множинної):", accuracy_score(y_test_multi, y_pred_multi))
