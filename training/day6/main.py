# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Importing the dataset
# Load the dataset into a Pandas DataFrame and extract features (X) and target (y).
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values  # Selecting Age and EstimatedSalary columns as features
y = dataset.iloc[:, 4].values      # Selecting the Purchased column as the target variable

# Step 2: Splitting the dataset into Training and Test sets
# Use sklearn's model_selection module to split the data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 3: Feature Scaling
# Standardize the features using StandardScaler for better performance of the Logistic Regression model.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit to the training data and transform it
X_test = sc.transform(X_test)        # Use the same transformation on the test data

# Step 4: Fitting Logistic Regression to the Training set
# Train the Logistic Regression model using the scaled training data.
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 5: Predicting the Test set results
# Use the trained model to predict outcomes for the test set.
y_pred = classifier.predict(X_test)

# Step 6: Making the Confusion Matrix
# Evaluate the performance of the model using a confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Printing the confusion matrix for evaluation
print("Confusion Matrix:")
print(cm)
