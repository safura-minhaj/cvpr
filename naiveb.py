import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Load the dataset
digits = load_digits()
X = digits.data    # Flattened images (64 features per image)
y = digits.target  # Labels (digits 0-9)

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Make Predictions and Print Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))