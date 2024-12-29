import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Handle missing values in the dataset using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Print after handling missing values
print("X after imputation:")
print(X)

# Encode categorical data (Country column)
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Print after label encoding (Country column)
print("\nX after Label Encoding (Country column):")
print(X)

# Apply One-Hot Encoding to the Country column
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# Print after One-Hot Encoding
print("\nX after One-Hot Encoding:")
print(X)

# Encode the target variable (Purchased column)
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Print target variable after Label Encoding
print("\nY after Label Encoding (Purchased column):")
print(Y)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Print training and testing sets
print("\nTraining Data (X_train):")
print(X_train)
print("\nTesting Data (X_test):")
print(X_test)

# Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Print scaled data
print("\nScaled Training Data (X_train):")
print(X_train)
print("\nScaled Testing Data (X_test):")
print(X_test)
