import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivatives for backpropagation
def relu_derivative(x):
    return (x > 0).astype(float)

class CircleNN:
    def __init__(self, input_size=2, hidden_size=8, output_size=1, learning_rate=0.1):
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # Output probability
        return self.a2

    def backward(self, X, Y, output):
        """Backward pass (gradient descent)"""
        m = X.shape[0]

        # Compute gradients
        dz2 = output - Y  # Error in output
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, Y, epochs=5000):
        """Train the network"""
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output)
            if i % 500 == 0:
                loss = -np.mean(Y * np.log(output + 1e-9) + (1 - Y) * np.log(1 - output + 1e-9))
                print(f"Epoch {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """Predict class (0 or 1)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


import matplotlib.pyplot as plt

# Generate dataset: Random (x, y) points
np.random.seed(42)
X = np.random.uniform(-1.5, 1.5, (1000, 2))
print("Input x: ", X)
Y = (X[:, 0]**2 + X[:, 1]**2 <= 1).astype(int).reshape(-1, 1)  # Unit circle condition
print("Output y: ", Y)

# Initialize neural network
nn = CircleNN(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1)

# Train model
nn.train(X, Y, epochs=5000)

# Test Predictions
preds = nn.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=preds.flatten(), cmap="coolwarm", alpha=0.5)
plt.title("Neural Network Learning the Unit Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
