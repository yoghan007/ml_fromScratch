"""
Machine Learning from Scratch: Utilities module
Includes data generation, scaling, splitting, and evaluation metrics.
"""

import numpy as np

# ==========================================
# Data Preprocessing & Utilities
# ==========================================

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean


def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split arrays or matrices into random train and test subsets."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_set_size = int(n_samples * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ==========================================
# Synthetic Data Generators
# ==========================================

def make_regression(n_samples=100, n_features=1, noise=10.0, random_state=42):
    """Generate a random regression problem."""
    np.random.seed(random_state)
    X = np.random.uniform(-5, 5, size=(n_samples, n_features))
    true_weights = np.random.uniform(2, 5, size=(n_features, 1))
    true_bias = 3.5
    
    y = X @ true_weights + true_bias + np.random.normal(0, noise, size=(n_samples, 1))
    
    if n_features == 1:
        return X, y.flatten(), true_weights.flatten(), true_bias
    return X, y, true_weights, true_bias


def make_classification(n_samples=100, n_features=2, noise=0.2, random_state=42):
    """Generate a 2D binary classification problem."""
    np.random.seed(random_state)
    n_class = n_samples // 2
    
    # Class 0 centered around (-1, -1)
    X0 = np.random.randn(n_class, n_features) * noise + np.array([-1.5, -1.5])
    y0 = np.zeros(n_class)
    
    # Class 1 centered around (1.5, 1.5)
    X1 = np.random.randn(n_samples - n_class, n_features) * noise + np.array([1.5, 1.5])
    y1 = np.ones(n_samples - n_class)
    
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    return X[indices], y[indices]


# ==========================================
# Evaluation Metrics
# ==========================================

def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """Calculate R^2 (coefficient of determination) regression score function."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - (ss_res / ss_tot)


def accuracy_score(y_true, y_pred):
    """Calculate Accuracy score for classification."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    """Calculate Precision score for binary classification."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(y_true, y_pred):
    """Calculate Recall score for binary classification."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    """Calculate F1 score for binary classification."""
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def confusion_matrix(y_true, y_pred):
    """Compute confusion matrix to evaluate the accuracy of a classification."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return np.array([[tn, fp],
                     [fn, tp]])
