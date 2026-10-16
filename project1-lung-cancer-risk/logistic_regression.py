import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda
        self.theta = None
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_samples, n_features = X.shape

        self.theta = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for epoch in range(self.num_epochs):
            perm = np.random.permutation(n_samples)
            X_shuf = X[perm]
            Y_shuf = Y[perm]

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                Xb = X_shuf[start:end]
                yb = Y_shuf[start:end]
                m = Xb.shape[0]

                logits = Xb.dot(self.theta) + self.bias
                probs = self.sigmoid(logits)

                error = probs - yb
                grad_theta = (Xb.T.dot(error)) / m
                if self.regularization_lambda != 0:
                    grad_theta = grad_theta + self.regularization_lambda * self.theta

                grad_bias = np.mean(error)
                self.theta -= self.learning_rate * grad_theta
                self.bias -= self.learning_rate * grad_bias



    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_samples = X.shape[0]
        logits = X.dot(self.theta) + self.bias
        probs = self.sigmoid(logits)
        error = probs - Y
        grad_theta = (X.T.dot(error)) / n_samples + self.regularization_lambda * self.theta
        grad_bias = np.mean(error)
        return grad_theta, grad_bias

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        X = np.asarray(X, dtype=float)
        logits = X.dot(self.theta) + self.bias
        probs = self.sigmoid(logits)
        return probs

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    