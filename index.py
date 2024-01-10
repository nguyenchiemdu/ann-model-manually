import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        # Backward propagation
        error = y - output

        # Output layer gradients
        output_delta = error * self.sigmoid_derivative(output)
        hidden_output_error = output_delta.dot(self.weights_hidden_output.T)

        # Hidden layer gradients
        hidden_output_delta = hidden_output_error * self.sigmoid_derivative(
            self.hidden_output
        )

        # Update weights and biases
        self.weights_hidden_output += (
            self.hidden_output.T.dot(output_delta) * self.learning_rate
        )
        self.bias_output += (
            np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        )

        self.weights_input_hidden += X.T.dot(hidden_output_delta) * self.learning_rate
        self.bias_hidden += (
            np.sum(hidden_output_delta, axis=0, keepdims=True) * self.learning_rate
        )

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            self.backward(X, y, output)

            # Print the loss every 1000 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")


# Example usage:
# Assuming X is your input data and y is your target output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train the neural network
input_size = X.shape[1]
hidden_size = 100
output_size = 1
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs=10000)

# Make predictions
predictions = nn.forward(X)
print("Final Predictions:")
print(predictions)
