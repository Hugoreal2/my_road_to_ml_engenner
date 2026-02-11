# Day 15 - Naive Bayes Classifier

## What is Naive Bayes?

Naive Bayes is a probabilistic machine learning algorithm based on **Bayes' Theorem**. It's called "naive" because it assumes that all features are independent of each other, which simplifies the calculations.

## Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Where:
- $P(A|B)$ = Posterior probability (probability of A given B)
- $P(B|A)$ = Likelihood (probability of B given A)
- $P(A)$ = Prior probability of A
- $P(B)$ = Prior probability of B

## Types of Naive Bayes Classifiers

1. **Gaussian Naive Bayes**: Used when features follow a normal distribution
2. **Multinomial Naive Bayes**: Used for discrete counts (e.g., word counts in text)
3. **Bernoulli Naive Bayes**: Used for binary/boolean features

## Advantages

- Simple and easy to implement
- Fast training and prediction
- Works well with high-dimensional data
- Performs well even with small training datasets
- Not sensitive to irrelevant features
- Good for text classification and spam filtering

## Disadvantages

- Assumes feature independence (which is rarely true in real life)
- If a categorical variable has a category in test data that wasn't observed in training data, the model will assign 0 probability (can be solved with smoothing)
- Can be outperformed by more complex models

## When to Use Naive Bayes?

- Text classification (spam detection, sentiment analysis)
- Real-time prediction (fast algorithm)
- Multi-class prediction
- When features are independent
- When you have limited training data

## Implementation Steps

1. **Import libraries and dataset**
2. **Split data** into training and test sets
3. **Feature scaling** (optional but recommended)
4. **Train the Gaussian Naive Bayes classifier**
5. **Make predictions** on test set
6. **Evaluate performance** using confusion matrix and accuracy

## Python Implementation

```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```

## Key Points

- Naive Bayes is a **probabilistic classifier**
- It calculates the probability of each class and selects the class with highest probability
- Despite its simplicity and "naive" assumption, it often performs surprisingly well
- Particularly effective for **text classification** tasks
- Can handle both binary and multi-class classification problems
