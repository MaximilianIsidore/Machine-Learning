import numpy as np
import matplotlib.pyplot as plt

# Cost function
def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Gradient Descent
def run_gradient_descent_feng(X, y, iterations, alpha):
    m = len(y)
    w = np.zeros(X.shape[1])  # Initialize weights (one per feature)
    b = 0  # Initialize bias
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(w) + b
        # Compute gradients
        dw = (1/m) * X.T.dot(predictions - y)
        db = (1/m) * np.sum(predictions - y)
        
        # Update weights and bias
        w -= alpha * dw
        b -= alpha * db
        
        # Record the cost (for monitoring)
        cost_history.append(compute_cost(X, y, w, b))
        
        # Optionally print the cost every 10000 iterations
        if i % 10000 == 0:
            print(f"Iteration {i}, Cost: {cost_history[-1]}")
            #print("dw shape: ", dw)
    
    return w, b

# Normalization
def zscore_normalize_features(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

# Data
x = np.arange(0, 20, 1)
y = x**2
X = np.c_[x, x**2, x**3]

print("Input X:", x)
print("Output Y:", y)

print("Input after feature Engineering: ", X)

print(X.std(axis=0))

# Normalize features
X = zscore_normalize_features(X)
print("Normalized X: ", X)

# Run gradient descent
model_w, model_b = run_gradient_descent_feng(X, y, iterations=20000, alpha=1e-1)

print("Weight :", model_w)
print("Bias :",model_b)

# Compute predictions
y_pred = X.dot(model_w) + model_b  # Use the learned weights and bias to make predictions

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Actual Values (y = x^2)", color='blue', marker='o')
plt.plot(x, y_pred, label="Predictions", color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
