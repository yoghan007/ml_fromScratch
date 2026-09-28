"""
Main Execution Script for Machine Learning from Scratch
Trains Linear & Logistic Regression models, evaluates metrics, and generates visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.utils import (
    make_regression,
    make_classification,
    train_test_split,
    StandardScaler,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.linear_regression import LinearRegression
from src.logistic_regression import LogisticRegression
from visualize import plot_linear_regression, plot_logistic_regression


def run_linear_regression_demo():
    print("\n" + "=" * 60)
    print(" 1. DEMO: LINEAR REGRESSION FROM SCRATCH")
    print("=" * 60)

    # 1. Generate Synthetic Dataset
    X, y, true_weights, true_bias = make_regression(n_samples=150, n_features=1, noise=4.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train via Gradient Descent
    print("\n[+] Training Linear Regression via Gradient Descent...")
    model_gd = LinearRegression(learning_rate=0.05, n_iterations=500, method="gradient_descent", l2_penalty=0.01)
    model_gd.fit(X_train_scaled, y_train)

    y_pred_gd = model_gd.predict(X_test_scaled)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    rmse_gd = root_mean_squared_error(y_test, y_pred_gd)
    r2_gd = r2_score(y_test, y_pred_gd)

    print(f"    - Final Training Cost: {model_gd.cost_history[-1]:.4f}")
    print(f"    - Learned Weight (w):  {model_gd.weights.item():.4f}")
    print(f"    - Learned Bias (b):    {model_gd.bias:.4f}")
    print(f"    - Test MSE:            {mse_gd:.4f}")
    print(f"    - Test RMSE:           {rmse_gd:.4f}")
    print(f"    - Test R^2 Score:      {r2_gd:.4f}")

    # 3. Train via Normal Equation (Closed-form solution)
    print("\n[+] Training Linear Regression via Normal Equation (Analytical Solution)...")
    model_ne = LinearRegression(method="normal_equation")
    model_ne.fit(X_train_scaled, y_train)

    y_pred_ne = model_ne.predict(X_test_scaled)
    r2_ne = r2_score(y_test, y_pred_ne)
    print(f"    - Analytical Weight (w): {model_ne.weights.item():.4f}")
    print(f"    - Analytical Bias (b):   {model_ne.bias:.4f}")
    print(f"    - Test R^2 Score:        {r2_ne:.4f}")

    # 4. Save Plots
    fig = plot_linear_regression(X_train_scaled, y_train, model_gd, title="Linear Regression: Fitted Line & Loss Curve")
    save_path = "linear_regression_results.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Saved plot to '{save_path}'")


def run_logistic_regression_demo():
    print("\n" + "=" * 60)
    print(" 2. DEMO: LOGISTIC REGRESSION FROM SCRATCH")
    print("=" * 60)

    # 1. Generate Synthetic Binary Classification Dataset
    X, y = make_classification(n_samples=200, n_features=2, noise=0.35, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Train Logistic Regression
    print("\n[+] Training Logistic Regression via Gradient Descent...")
    model_log = LogisticRegression(learning_rate=0.1, n_iterations=800, l2_penalty=0.01)
    model_log.fit(X_train_scaled, y_train)

    y_pred = model_log.predict(X_test_scaled)
    y_probs = model_log.predict_proba(X_test_scaled)

    # 3. Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"    - Final Training BCE Loss: {model_log.cost_history[-1]:.4f}")
    print(f"    - Learned Weights (w):      {model_log.weights.flatten().round(4)}")
    print(f"    - Learned Bias (b):         {model_log.bias:.4f}")
    print(f"    - Test Accuracy:            {acc * 100:.2f}%")
    print(f"    - Test Precision:           {prec * 100:.2f}%")
    print(f"    - Test Recall:              {rec * 100:.2f}%")
    print(f"    - Test F1 Score:            {f1:.4f}")
    print("    - Confusion Matrix (TN, FP / FN, TP):")
    print(f"        [[{cm[0, 0]:2d}, {cm[0, 1]:2d}],")
    print(f"         [{cm[1, 0]:2d}, {cm[1, 1]:2d}]]")

    # 4. Save Plots
    fig = plot_logistic_regression(X_train_scaled, y_train, model_log, title="Logistic Regression: Decision Boundary & Probability Heatmap")
    save_path = "logistic_regression_results.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[OK] Saved plot to '{save_path}'")


if __name__ == "__main__":
    print("\n============================================================")
    print("      MACHINE LEARNING FROM SCRATCH ENGINE                  ")
    print("============================================================")
    run_linear_regression_demo()
    run_logistic_regression_demo()
    print("\n" + "=" * 60)
    print(" All demonstrations completed successfully!")
    print("=" * 60 + "\n")
