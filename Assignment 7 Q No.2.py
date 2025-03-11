import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 0: Read data into a pandas dataframe
data = pd.read_csv("suv.csv")

# Step 1: Select features (Age, EstimatedSalary) and target (Purchased)
X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]

# Step 2: Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train decision tree classifier with entropy criterion
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_entropy.fit(X_train, y_train)

# Predict and evaluate (Entropy criterion)
y_pred_entropy = dt_entropy.predict(X_test)
print("Confusion Matrix (Entropy Criterion):\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report (Entropy Criterion):\n", classification_report(y_test, y_pred_entropy))

# Step 5: Train decision tree classifier with gini criterion
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
dt_gini.fit(X_train, y_train)

# Predict and evaluate (Gini criterion)
y_pred_gini = dt_gini.predict(X_test)
print("Confusion Matrix (Gini Criterion):\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report (Gini Criterion):\n", classification_report(y_test, y_pred_gini))

# Step 6: Compare model performances
print("Comparison:")
print("- Entropy criterion tends to create deeper trees, which might capture more complexity.")
print("- Gini criterion is computationally simpler and often leads to similar performance.")
print("- Evaluate accuracy, precision, recall, and F1-score to decide the better approach.")
