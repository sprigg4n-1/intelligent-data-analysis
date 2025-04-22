import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from scipy.stats import uniform
from sklearn.utils import shuffle

df = pd.read_csv("../tested.csv", encoding="latin1")

label_encoder = LabelEncoder()
scaler = StandardScaler()

df_copy = df.copy()

print(df_copy.isnull().sum())

for col in df_copy.select_dtypes(include=['float64', 'int64']).columns:
    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

for col in df_copy.select_dtypes(include=['object']).columns:
    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

print(df_copy.isnull().sum())


df_copy[['Age', 'Fare']] = scaler.fit_transform(df_copy[['Age', 'Fare']])

df_copy['Sex'] = label_encoder.fit_transform(df_copy['Sex'])  
df_copy['Embarked'] = label_encoder.fit_transform(df_copy['Embarked'])  

X = df_copy[['Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df_copy['Survived']

# Задаємо перемішування (shuffle) для кожного методу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = svm.SVC(kernel='linear', random_state=100)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

# Оцінка за допомогою Leave-One-Out Cross-Validation (LOO CV)
# Без перемішування
loo = LeaveOneOut()
loo_accuracies_no_shuffle = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    loo_accuracies_no_shuffle.append(accuracy_score(y_test, y_pred))

print(f"LeaveOneOut точність без перемішування: {np.mean(loo_accuracies_no_shuffle):.4f}")

# З перемішуванням
loo_accuracies_with_shuffle = []
X_shuffled, y_shuffled = shuffle(X, y, random_state=100)  # Перемішуємо дані
for train_idx, test_idx in loo.split(X_shuffled):
    X_train, X_test = X_shuffled.iloc[train_idx], X_shuffled.iloc[test_idx]
    y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]
    
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    loo_accuracies_with_shuffle.append(accuracy_score(y_test, y_pred))

print(f"LeaveOneOut точність з перемішуванням: {np.mean(loo_accuracies_with_shuffle):.4f}")

# Оцінка за допомогою K-Fold Cross-Validation (5-фолдова крос-валідація)
# Без перемішування
cv = StratifiedKFold(n_splits=5)
cv_accuracies_no_shuffle = []
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    cv_accuracies_no_shuffle.append(accuracy_score(y_test, y_pred))

print(f"K-Fold CV точність без перемішування: {np.mean(cv_accuracies_no_shuffle):.4f}")

# З перемішуванням
cv_accuracies_with_shuffle = []
X_shuffled, y_shuffled = shuffle(X, y, random_state=100)  # Перемішуємо дані
for train_idx, test_idx in cv.split(X_shuffled, y_shuffled):
    X_train, X_test = X_shuffled.iloc[train_idx], X_shuffled.iloc[test_idx]
    y_train, y_test = y_shuffled.iloc[train_idx], y_shuffled.iloc[test_idx]
    
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)
    cv_accuracies_with_shuffle.append(accuracy_score(y_test, y_pred))

print(f"K-Fold CV точність з перемішуванням: {np.mean(cv_accuracies_with_shuffle):.4f}")


# Сітковий пошук (GridSearchCV)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Без перемішування
grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Найкращі параметри (GridSearch без перемішування):", grid_search.best_params_)
y_pred_best = grid_search.best_estimator_.predict(X_test)
print(f"Точність GridSearchCV без перемішування: {accuracy_score(y_test, y_pred_best):.4f}")

# З перемішуванням
grid_search_shuffled = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
X_shuffled, y_shuffled = shuffle(X, y, random_state=100)  # Перемішуємо дані
grid_search_shuffled.fit(X_shuffled, y_shuffled)
print("Найкращі параметри (GridSearch з перемішуванням):", grid_search_shuffled.best_params_)
y_pred_best_shuffled = grid_search_shuffled.best_estimator_.predict(X_test)
print(f"Точність GridSearchCV з перемішуванням: {accuracy_score(y_test, y_pred_best_shuffled):.4f}")

# Випадковий пошук (RandomizedSearchCV)
param_dist = {
    'C': uniform(0.1, 100),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Без перемішування
random_search = RandomizedSearchCV(svm.SVC(), param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=100)
random_search.fit(X_train, y_train)
print("Найкращі параметри (RandomizedSearch без перемішування):", random_search.best_params_)
y_pred_random = random_search.best_estimator_.predict(X_test)
print(f"Точність RandomizedSearch без перемішування: {accuracy_score(y_test, y_pred_random):.4f}")

# З перемішуванням
random_search_shuffled = RandomizedSearchCV(svm.SVC(), param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=100)
X_shuffled, y_shuffled = shuffle(X, y, random_state=100)  # Перемішуємо дані
random_search_shuffled.fit(X_shuffled, y_shuffled)
print("Найкращі параметри (RandomizedSearch з перемішуванням):", random_search_shuffled.best_params_)
y_pred_random_shuffled = random_search_shuffled.best_estimator_.predict(X_test)
print(f"Точність RandomizedSearch з перемішуванням: {accuracy_score(y_test, y_pred_random_shuffled):.4f}")
