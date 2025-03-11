import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read data into a pandas dataframe
data = pd.read_csv("data_banknote_authentication.csv")

# Step 1: Pick 'class' as target variable y and other columns as feature variables X
X = data.drop(columns=["class"])
y = data["class"]

# Step 2: Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Step 3: Train SVM with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

# Step 4: Predict and evaluate for linear kernel
y_pred_linear = svm_linear.predict(X_test)
print("Confusion Matrix (Linear Kernel):\n", confusion_matrix(y_test, y_pred_linear))
print("Classification Report (Linear Kernel):\n", classification_report(y_test, y_pred_linear))

# Step 5: Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

# Predict and evaluate for RBF kernel
y_pred_rbf = svm_rbf.predict(X_test)
print("Confusion Matrix (RBF Kernel):\n", confusion_matrix(y_test, y_pred_rbf))
print("Classification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))

# Step 6: Compare the two models
print("Comparison:")
print("- The linear kernel is often preferable when data is linearly separable.")
print("- The RBF kernel is useful when the data has complex, non-linear decision boundaries.")
print("- Performance can be compared using precision, recall, and F1-score from the classification reports.")
