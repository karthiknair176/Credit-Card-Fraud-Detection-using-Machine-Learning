# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('creditcard.c sv')
print("Dataset shape:", data.shape)
print(data['Class'].value_counts())

# Check for imbalance
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0]
print("Fraud transactions:", len(fraud))
print("Non-fraud transactions:", len(non_fraud))

# Downsample normal transactions to balance the dataset
non_fraud_sample = non_fraud.sample(len(fraud))
balanced_data = pd.concat([fraud, non_fraud_sample])

# Split features and labels
X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----- KNN Model -----
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# ----- SVM Model -----
svm = SVC(kernel='rbf', C=1, gamma='auto')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# ----- Evaluation -----
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title("KNN Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title("SVM Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()
