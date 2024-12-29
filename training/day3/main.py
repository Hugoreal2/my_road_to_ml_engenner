# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split  # Corrected deprecated import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

# Step 1: Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # Independent variables (all columns except the last)
Y = dataset.iloc[:, -1].values   # Dependent variable (last column)

# Step 2: Encoding Categorical data
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])  # Encoding the categorical column (e.g., State)

# Applying one-hot encoding to the encoded categorical column
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# Step 3: Avoiding the Dummy Variable Trap
# Removing the first dummy variable column to avoid multicollinearity
X = X[:, 1:]

# Step 4: Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Step 5: Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Step 6: Predicting the Test set results
y_pred = regressor.predict(X_test)

# Displaying the predictions
print("Predicted values for the test set:")
print(y_pred)

# Displaying the actual values for comparison
print("Actual values of the test set:")
print(Y_test)
