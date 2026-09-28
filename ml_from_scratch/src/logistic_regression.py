"""
Logistic Regression from Scratch using Pure NumPy

Hypothesis / Activation:
    z = X * w + b
    a = sigmoid(z) = 1 / (1 + exp(-z))

Cost Function (Binary Cross-Entropy / Log Loss):
    J(w, b) = - (1 / m) * sum( y * log(a) + (1 - y) * log(1 - a) ) + (l2 / 2m) * ||w||_2^2 + (l1 / m) * ||w||_1

Gradients:
    dJ/dw = (1 / m) * X^T * (a - y) + (l2 / m) * w + (l1 / m) * sign(w)
    dJ/db = (1 / m) * sum(a - y)
"""

import numpy as np

def sigmoid(z):
    """
    Numerically stable Sigmoid Activation Function.
    sigma(z) = 1 / (1 + e^(-z))
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegression:
    def __init__(self, learning_rate=0.05, n_iterations=1000, 
                 l1_penalty=0.0, l2_penalty=0.0, threshold=0.5, tol=1e-6, verbose=False):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate step size for gradient descent.
        n_iterations : int
            Maximum number of gradient descent epochs.
        l1_penalty : float
            L1 regularization coefficient (Lasso).
        l2_penalty : float
            L2 regularization coefficient (Ridge).
        threshold : float
            Decision boundary probability threshold (default 0.5).
        tol : float
            Tolerance for early stopping based on cost loss change.
        verbose : bool
            Whether to print cost during training iterations.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.threshold = threshold
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _compute_loss(self, X, y, a):
        """
        Compute Binary Cross Entropy Loss (Log Loss) with L1/L2 regularization.
        """
        m = X.shape[0]
        eps = 1e-15  # Avoid log(0)
        a = np.clip(a, eps, 1 - eps)
        
        bce_loss = - (1 / m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        
        l2_loss = (self.l2_penalty / (2 * m)) * np.sum(self.weights ** 2) if self.l2_penalty > 0 else 0.0
        l1_loss = (self.l1_penalty / m) * np.sum(np.abs(self.weights)) if self.l1_penalty > 0 else 0.0
        
        return bce_loss + l2_loss + l1_loss

    def fit(self, X, y):
        """
        Fit Logistic Regression model to training data (X, y).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m, n_features = X.shape

        # Initialize weights randomly near zero, bias to zero
        np.random.seed(42)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.n_iterations):
            # Forward pass: z = X * w + b, a = sigmoid(z)
            z = X @ self.weights + self.bias
            a = sigmoid(z)

            # Compute error
            error = a - y

            # Gradients
            dw = (1 / m) * (X.T @ error)
            db = (1 / m) * np.sum(error)

            # Apply regularization gradients
            if self.l2_penalty > 0:
                dw += (self.l2_penalty / m) * self.weights
            if self.l1_penalty > 0:
                dw += (self.l1_penalty / m) * np.sign(self.weights)

            # Update weights & bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Compute and record cost
            cost = self._compute_loss(X, y, a)
            self.cost_history.append(cost)

            if self.verbose and (i % max(1, self.n_iterations // 10) == 0 or i == self.n_iterations - 1):
                print(f"Epoch {i:5d} | Binary Cross-Entropy Loss: {cost:.6f}")

            # Check convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at epoch {i} with loss {cost:.6f}")
                break

        return self

    def predict_proba(self, X):
        """
        Predict probability estimates for positive class P(y=1|X).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        z = X @ self.weights + self.bias
        probs = sigmoid(z)
        return probs.flatten() if probs.shape[1] == 1 else probs

    def predict(self, X):
        """
        Predict binary class labels (0 or 1) using threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
