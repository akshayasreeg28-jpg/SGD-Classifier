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
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction][0])

plt.scatter(X[:,0], X[:,1], c=y)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification")

plt.show()
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Akshaya Sree G
RegisterNumber: 212225230011 
*/
```

## Output:
<img width="568" height="455" alt="WhatsApp Image 2026-05-25 at 10 28 56" src="https://github.com/user-attachments/assets/1c067cf1-7e5c-4f94-8e36-0a709c4b91a3" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
