import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Avoid numpy printing warning
np.set_printoptions(threshold=np.inf)

# Load dataset
dataset = pd.read_csv(r"C:\Users\rosha\OneDrive\Desktop\Gitesh\NIT\TOPICS\ML\Housing Price Prediction\House_data.csv")
space = dataset['sqft_living']
price = dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=1/3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# Predictions
pred = regressor.predict(xtest)

# Training set visualization
plt.scatter(xtrain, ytrain, color='red')
plt.plot(np.sort(xtrain, axis=0), regressor.predict(np.sort(xtrain, axis=0)), color='blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space in sqft")
plt.ylabel("Price")
plt.show()

# Test set visualization
plt.scatter(xtest, ytest, color='red')
plt.plot(np.sort(xtest, axis=0), regressor.predict(np.sort(xtest, axis=0)), color='blue')
plt.title("Visuals for Testing Dataset")
plt.xlabel("Space in sqft")
plt.ylabel("Price")
plt.show()