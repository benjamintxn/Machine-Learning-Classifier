# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.

import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size=25, hidden_size=10, output_size=4, learning_rate=0.01):
        # Initialize parameters
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def reset(self):
        # Reinitialize parameters
        self.__init__()

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / exp_Z.sum(axis=1, keepdims=True)

    def forward_prop(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.ReLU(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self.b2.shape[1]))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    def backward_prop(self, X, Y):
        m = X.shape[0]
        one_hot_Y = self.one_hot(Y)

        dZ2 = self.A2 - one_hot_Y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * self.ReLU_deriv(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def get_predictions(self, A2):
        return np.argmax(A2, axis=1)

    def get_accuracy(self, predictions, Y):
        return np.mean(predictions == Y)

    def gradient_descent(self, X, Y, iterations=1000, batch_size=32):
        m = X.shape[0]
        for _ in range(iterations):
            idx = np.random.choice(m, min(batch_size, m), replace=False)
            X_batch = X[idx]
            Y_batch = Y[idx]

            self.forward_prop(X_batch)
            dW1, db1, dW2, db2 = self.backward_prop(X_batch, Y_batch)
            self.update_params(dW1, db1, dW2, db2)

class Classifier:
    def __init__(self):
        self.model = NeuralNetwork(learning_rate=0.01)
        self.k_folds = 5  # Number of folds for cross-validation

    def reset(self):
        self.model.reset()

    def fit(self, data, target):
        X = np.array(data)
        Y = np.array(target)

        # Perform 5-fold cross-validation
        fold_size = len(X) // self.k_folds
        accuracies = []

        for fold in range(self.k_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size

            X_val = X[val_start:val_end]
            Y_val = Y[val_start:val_end]

            X_train = np.concatenate([X[:val_start], X[val_end:]])
            Y_train = np.concatenate([Y[:val_start], Y[val_end:]])

            self.model.gradient_descent(X_train, Y_train, iterations=500, batch_size=32)

            predictions = self.model.get_predictions(self.model.forward_prop(X_val))
            accuracy = self.model.get_accuracy(predictions, Y_val)
            accuracies.append(accuracy)

            print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

            # Debugging output: Print predicted vs actual moves for the validation set
            print(f"Fold {fold + 1} Predictions:")
            for i in range(len(X_val)):
                predicted_move = predictions[i]
                actual_move = Y_val[i]
                print(f"  Example {i + 1}: Predicted Move = {predicted_move}, Actual Move = {actual_move}")

        print(f"Average Cross-Validation Accuracy: {np.mean(accuracies):.4f}")

        # Retrain on the full dataset
        self.model.gradient_descent(X, Y, iterations=500, batch_size=32)

    def predict(self, data, legal=None, epsilon=0.1):
        # Reshape the input data to match the expected format
        X = np.array(data).reshape(1, -1)

        # Get the output probabilities from the neural network
        probs = self.model.forward_prop(X)[0]

        if legal is not None:
            # Filter out 'Stop' and other unsupported actions
            legal_indices = [i for i in range(4) if i < len(legal)]
            legal_probs = np.full(4, -np.inf)  # Initialize all probabilities to -inf
            legal_probs[legal_indices] = probs[legal_indices]  # Assign valid probabilities
            chosen_action = np.argmax(legal_probs)

            return chosen_action

        # If no legal actions are provided, return the index of the highest probability action
        return np.argmax(probs)