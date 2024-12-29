import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Create the dataset
data = {
    'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4],
    'Price': [400000, 450000, 475000, 500000, 350000, 425000, 600000, 650000]
}

df = pd.DataFrame(data)

# Step 2: Prepare the features (X) and target (y)
X = df[['Size', 'Bedrooms']]  # Features: Size and Bedrooms
y = df['Price']  # Target: Price

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model (using Mean Squared Error for simplicity)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 7: Use the model to predict the price of a new house
new_house = np.array([[2000, 3]])  # Size = 2000 sqft, Bedrooms = 3
predicted_price = model.predict(new_house)
print(f'Predicted price for the new house: ${predicted_price[0]:,.2f}')
