"""
ML from Scratch - Core Package
"""
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .utils import (
    StandardScaler,
    train_test_split,
    make_regression,
    make_classification,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "StandardScaler",
    "train_test_split",
    "make_regression",
    "make_classification",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix"
]
