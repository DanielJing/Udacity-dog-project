import numpy as np

# Define sigmoid, the activation function 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of activation function 
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the backpropagation algorithm
def backpropagation(X, y, num_iterations, learning_rate):
    np.random.seed(1)
    weights = np.random.randn(X.shape[1], 1)

    for i in range(num_iterations):
        # Forward propagation
        z = np.dot(X, weights)
        y_pred = sigmoid(z)

        error = y_pred - y

        # Backward propagation
        d_weights = np.dot(X.T, error * sigmoid_derivative(z))

        weights -= learning_rate * d_weights

    return weights

X = np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[1], [0], [0], [1]])
weights = backpropagation(X, y, 100000, 0.1)
print(weights)
