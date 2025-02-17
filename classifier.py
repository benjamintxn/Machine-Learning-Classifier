import numpy as np
import random

class NeuralNetwork:
    def __init__(self, input_size=25, hidden_size=100, output_size=4, learning_rate=0.1, dropout_rate=0.2, reg_lambda=0.001):
        """Initializes the neural network with given hyperparameters."""
        self.__dict__.update({"input_size": input_size, "hidden_size": hidden_size, "output_size": output_size, "initial_learning_rate": learning_rate, "learning_rate": learning_rate, "dropout_rate": dropout_rate, "reg_lambda": reg_lambda, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "t": 0})
        self._initialize_weights()

    def reset(self):
        self.t = 0
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization and set up Adam optimizer variables."""
        self.W1, self.b1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size), np.zeros((1, self.hidden_size))
        self.W2, self.b2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size), np.zeros((1, self.output_size))

        # Initialize Adam moment variables (velocity & squared gradients)
        self.v_W1, self.v_b1, self.v_W2, self.v_b2 = [np.zeros_like(var) for var in (self.W1, self.b1, self.W2, self.b2)]
        self.s_W1, self.s_b1, self.s_W2, self.s_b2 = [np.zeros_like(var) for var in (self.W1, self.b1, self.W2, self.b2)]

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward_prop(self, X, training=True):
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.ReLU(self.Z1)
        # Apply dropout during training
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.A1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.A1 *= self.dropout_mask
        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, self.output_size))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y

    def compute_loss(self, Y, A2):
        m = Y.shape[0]
        log_likelihood = -np.log(A2[np.arange(m), Y] + 1e-8)
        data_loss = np.sum(log_likelihood) / m
        # Add L2 regularization for W1 and W2
        reg_loss = (self.reg_lambda / (2 * m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return data_loss + reg_loss

    def backward_prop(self, X, Y, training=True):
        """Performs backpropagation to compute weight gradients."""
        m = X.shape[0]
        dZ2 = self.A2 - self.one_hot(Y)  # Output layer error
        dW2, db2 = np.dot(self.A1.T, dZ2) / m + (self.reg_lambda / m) * self.W2, np.sum(dZ2, axis=0, keepdims=True) / m

        dZ1 = np.dot(dZ2, self.W2.T) * (self.Z1 > 0)  # ReLU derivative
        if training and self.dropout_rate > 0:
            dZ1 *= self.dropout_mask  # Apply dropout during backprop

        dW1, db1 = np.dot(X.T, dZ1) / m + (self.reg_lambda / m) * self.W1, np.sum(dZ1, axis=0, keepdims=True) / m
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2):
        """ Update weights using the Adam optimization algorithm. """
        self.t += 1  # Increment time step

        params = {
            "W1": (self.W1, dW1, self.v_W1, self.s_W1),
            "b1": (self.b1, db1, self.v_b1, self.s_b1),
            "W2": (self.W2, dW2, self.v_W2, self.s_W2),
            "b2": (self.b2, db2, self.v_b2, self.s_b2),
        }

        for key, (param, dparam, v, s) in params.items():
            # Compute moving averages
            v[:] = self.beta1 * v + (1 - self.beta1) * dparam
            s[:] = self.beta2 * s + (1 - self.beta2) * (dparam ** 2)

            # Bias correction
            v_corr = v / (1 - self.beta1 ** self.t)
            s_corr = s / (1 - self.beta2 ** self.t)

            # Update parameters
            param -= self.learning_rate * v_corr / (np.sqrt(s_corr) + self.epsilon)

    def train(self, X, Y, epochs=250, batch_size=32, decay_rate=0.95, decay_steps=100, 
              X_val=None, Y_val=None, patience=20):
        m = X.shape[0]
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Shuffle training data each epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]
            epoch_loss = 0.0
            num_batches = int(np.ceil(m / batch_size))
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, m)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                A2 = self.forward_prop(X_batch, training=True)
                batch_loss = self.compute_loss(Y_batch, A2)
                epoch_loss += batch_loss * (end - start)
                dW1, db1, dW2, db2 = self.backward_prop(X_batch, Y_batch, training=True)
                self.update_params(dW1, db1, dW2, db2)
            epoch_loss /= m

            # Apply learning rate decay every decay_steps epochs
            self.learning_rate = self.initial_learning_rate * (decay_rate ** (epoch / decay_steps))
            
            # Compute training accuracy
            train_preds = np.argmax(self.forward_prop(X, training=False), axis=1)
            train_accuracy = np.mean(train_preds == Y)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, LR: {self.learning_rate:.6f}")

            # If validation data is provided, compute validation loss for early stopping
            if X_val is not None and Y_val is not None:
                val_A2 = self.forward_prop(X_val, training=False)
                val_loss = self.compute_loss(Y_val, val_A2)
                print(f"  Validation Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print("Early stopping triggered.")
                        break

    def predict(self, X):
        A2 = self.forward_prop(X, training=False)
        return np.argmax(A2, axis=1)

class Classifier:
    def __init__(self):
        self.model = NeuralNetwork(learning_rate=0.1)

    def reset(self):
        self.model.reset()

    def fit(self, data, target):
        X = np.array(data, dtype=float)
        Y = np.array(target, dtype=int)
        # Optionally, split a validation set from X and Y here for early stopping.
        self.model.train(X, Y, epochs=250, batch_size=32, decay_rate=0.95, decay_steps=100, X_val=None, Y_val=None, patience=20)

    def predict(self, data, legal=None, epsilon=0.1):
        X = np.array(data, dtype=float).reshape(1, -1)
        probs = self.model.forward_prop(X, training=False)[0]
        if legal is not None:
            legal_indices = [i for i in range(4) if i < len(legal)]
            legal_probs = np.full(4, -np.inf)
            legal_probs[legal_indices] = probs[legal_indices]
            return int(np.argmax(legal_probs))
        return int(np.argmax(probs))