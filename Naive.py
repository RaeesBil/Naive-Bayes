import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('adult.csv')

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop('income', axis=1))
X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:\n', confusion)
print('Classification Report:\n', report)

plt.figure(figsize=(10, 7))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['income'].classes_, yticklabels=label_encoders['income'].classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

feature1 = 'age'
feature2 = 'hours.per.week'

X_train_2d = X_train[[feature1, feature2]]
X_test_2d = X_test[[feature1, feature2]]
model.fit(X_train_2d, y_train)

x_min, x_max = X_test_2d[feature1].min() - 1, X_test_2d[feature1].max() + 1
y_min, y_max = X_test_2d[feature2].min() - 1, X_test_2d[feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X_test_2d[feature1], X_test_2d[feature2], c=y_test, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.title(f'Naive Bayes Decision Boundaries ({feature1} vs {feature2})')
plt.show()
