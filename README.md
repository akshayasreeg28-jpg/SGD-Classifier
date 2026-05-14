# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare the Dataset
Import the dataset, separate input features (X) and target labels (y), and split the data into training and testing sets.
2. Initialize the SGD Classifier Model
Create the SGD Classifier by setting parameters such as loss function, learning rate, maximum iterations, and random state.
3. Train the Model Using Stochastic Gradient Descent
Fit the classifier with the training data, where model parameters are updated iteratively for each training sample.
4. Predict and Evaluate the Model
Predict class labels for test data and evaluate the classifier performance using accuracy score, confusion matrix, or classification report.
 

## Program:
```
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
data = pd.read_csv("Placement_Data.csv")

# Separate Features and Target
X = data.drop(["status", "salary", "sl_no"], axis=1)
y = data["status"]

# One-Hot Encoding for Categorical Columns
X = pd.get_dummies(X, drop_first=True)

# Save Feature Names
feature_names = X.columns

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling (keep as DataFrame)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_names
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=feature_names
)

# Create SGD Classifier
model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

# Train Model
model.fit(X_train_scaled, y_train)

# Predict on Test Data
y_pred = model.predict(X_test_scaled)

# Accuracy & Reports
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#  Correct New Student Prediction

# Create empty student with all features = 0
new_student_dict = dict.fromkeys(feature_names, 0)

# Fill ONLY numerical features (example values)
new_student_dict['ssc_p'] = 67
new_student_dict['hsc_p'] = 91
new_student_dict['degree_p'] = 58
new_student_dict['etest_p'] = 88
new_student_dict['mba_p'] = 67

# Convert to DataFrame with same columns
new_student_df = pd.DataFrame([new_student_dict])

# Scale using same scaler
new_student_scaled = pd.DataFrame(
    scaler.transform(new_student_df),
    columns=feature_names
)

# Predict
pred = model.predict(new_student_scaled)
print("\nPredicted Status:", pred[0])

/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Akshaya Sree G
RegisterNumber: 212225230011 
*/
```

## Output:
<img width="967" height="730" alt="image" src="https://github.com/user-attachments/assets/e39aa5a8-3fe1-4a0e-94d9-7e2674faf5df" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
