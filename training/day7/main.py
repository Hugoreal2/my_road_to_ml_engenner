# Import necessary libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Sample dataset: [Height, Weight] and Gender
X = np.array([[150, 50], [160, 55], [170, 70], [180, 80], [175, 65], [155, 60]])  # Features (Height, Weight)
y = np.array(['F', 'F', 'M', 'M', 'M', 'F'])  # Labels (Gender)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize the K-NN classifier with K=3
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(X_train, y_train)

# New data point (Height=162 cm, Weight=58 kg)
new_data = np.array([[162, 58]])

# Predict the class for the new data point
prediction = classifier.predict(new_data)

print("Predicted Gender:", prediction[0])  # Output: 'F'
