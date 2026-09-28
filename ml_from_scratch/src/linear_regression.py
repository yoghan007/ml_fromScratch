"""
Linear Regression from Scratch using Pure NumPy

Hypothesis:
    y_hat = X * w + b

Cost Function (MSE + Regularization):
    J(w, b) = (1 / 2m) * sum((y_hat - y)^2) + (l2_reg / 2m) * ||w||_2^2 + (l1_reg / m) * ||w||_1

Gradients:
    dJ/dw = (1 / m) * X^T * (y_hat - y) + (l2_reg / m) * w + (l1_reg / m) * sign(w)
    dJ/db = (1 / m) * sum(y_hat - y)
"""

import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method="gradient_descent", 
                 l1_penalty=0.0, l2_penalty=0.0, batch_size=None, tol=1e-6, verbose=False):
        """
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent updates.
        n_iterations : int
            Maximum number of gradient descent iterations (epochs).
        method : str
            Optimization solver: 'gradient_descent' or 'normal_equation'.
        l1_penalty : float
            L1 regularization hyperparameter (Lasso).
        l2_penalty : float
            L2 regularization hyperparameter (Ridge).
        batch_size : int or None
            Size of mini-batches. If None, performs Batch Gradient Descent.
        tol : float
            Tolerance for early stopping based on loss changes.
        verbose : bool
            Whether to print loss progress during training.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.weight_history = []

    def _compute_cost(self, X, y, y_pred):
        """
        Compute Mean Squared Error with L1 and L2 regularization.
        J(w, b) = (1 / 2m) * sum((y_pred - y)^2) + (l2 / 2m) * ||w||_2^2 + (l1 / m) * ||w||_1
        """
        m = X.shape[0]
        mse_cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        
        l2_cost = (self.l2_penalty / (2 * m)) * np.sum(self.weights ** 2) if self.l2_penalty > 0 else 0.0
        l1_cost = (self.l1_penalty / m) * np.sum(np.abs(self.weights)) if self.l1_penalty > 0 else 0.0
        
        return mse_cost + l2_cost + l1_cost

    def fit(self, X, y):
        """
        Fit linear model to training data (X, y).
        """
        # Ensure 2D arrays
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        m, n_features = X.shape

        if self.method == "normal_equation":
            self._fit_normal_equation(X, y)
            return self

        # Initialize parameters: weights to small random numbers, bias to zero
        np.random.seed(42)
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0.0
        self.cost_history = []
        self.weight_history = []

        for i in range(self.n_iterations):
            if self.batch_size is None or self.batch_size >= m:
                # Batch Gradient Descent
                self._update_weights(X, y, m)
            else:
                # Mini-batch / Stochastic Gradient Descent
                indices = np.arange(m)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for start_idx in range(0, m, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, m)
                    X_b = X_shuffled[start_idx:end_idx]
                    y_b = y_shuffled[start_idx:end_idx]
                    self._update_weights(X_b, y_b, len(X_b))

            # Track total cost on whole dataset
            y_pred = X @ self.weights + self.bias
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
            self.weight_history.append(self.weights.copy())

            if self.verbose and (i % max(1, self.n_iterations // 10) == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i:5d} | Cost J(w, b): {cost:.6f}")

            # Check for convergence / early stopping
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {i} with cost {cost:.6f}")
                break

        return self

    def _update_weights(self, X_batch, y_batch, batch_m):
        """
        Perform vectorized gradient descent update step.
        """
        y_pred = X_batch @ self.weights + self.bias
        error = y_pred - y_batch

        # Gradient w.r.t weights: (1/m) * X^T * error + regularization
        dw = (1 / batch_m) * (X_batch.T @ error)
        
        if self.l2_penalty > 0:
            dw += (self.l2_penalty / batch_m) * self.weights
        if self.l1_penalty > 0:
            dw += (self.l1_penalty / batch_m) * np.sign(self.weights)

        # Gradient w.r.t bias: (1/m) * sum(error)
        db = (1 / batch_m) * np.sum(error)

        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def _fit_normal_equation(self, X, y):
        """
        Analytical solution using Normal Equation:
        w_full = (X_b^T * X_b + lambda * I)^(-1) * X_b^T * y
        where X_b includes a column of 1s for bias.
        """
        m = X.shape[0]
        # Add bias column (column of ones)
        X_b = np.hstack((np.ones((m, 1)), X))
        
        # Identity matrix for L2 Ridge Regularization (do not penalize bias term)
        n_col = X_b.shape[1]
        I = np.eye(n_col)
        I[0, 0] = 0.0

        # Solves (X_b^T * X_b + lambda * I) * params = X_b^T * y
        A = X_b.T @ X_b + self.l2_penalty * I
        b_vec = X_b.T @ y

        params = np.linalg.pinv(A) @ b_vec
        self.bias = params[0, 0]
        self.weights = params[1:]
        
        # Calculate final cost
        y_pred = X @ self.weights + self.bias
        cost = self._compute_cost(X, y, y_pred)
        self.cost_history = [cost]

    def predict(self, X):
        """
        Predict using the linear model.
        y_hat = X * w + b
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        predictions = X @ self.weights + self.bias
        return predictions.flatten() if predictions.shape[1] == 1 else predictions
