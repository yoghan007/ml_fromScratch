"""
Matplotlib Visualization Suite for ML from Scratch
Plots regression lines, decision boundaries, cost decay, and loss contours.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_linear_regression(X_train, y_train, model, title="Linear Regression from Scratch"):
    """
    Plot 1D Linear Regression scatter data, fitted line, and cost curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Fitted Line Plot
    axes[0].scatter(X_train, y_train, color="#3b82f6", alpha=0.7, label="Training Data Points")
    
    x_min, x_max = X_train.min() - 1, X_train.max() + 1
    x_line = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    y_line = model.predict(x_line)
    
    axes[0].plot(x_line, y_line, color="#ef4444", linewidth=2.5, 
                 label=f"Fitted Line: y = {model.weights.item():.2f}x + {model.bias:.2f}")
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Feature (X)")
    axes[0].set_ylabel("Target (y)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Loss Curve Plot
    axes[1].plot(model.cost_history, color="#10b981", linewidth=2)
    axes[1].set_title("Cost Function Decay J(w, b)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch Iterations")
    axes[1].set_ylabel("Mean Squared Error Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_logistic_regression(X_train, y_train, model, title="Logistic Regression Decision Boundary"):
    """
    Plot 2D Binary Classification data, decision boundary, and cost curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Decision Boundary Contour Plot
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid_points).reshape(xx.shape)
    
    # Probability surface heatmap
    contour = axes[0].contourf(xx, yy, probs, levels=20, cmap="coolwarm", alpha=0.6)
    fig.colorbar(contour, ax=axes[0], label="Probability P(y=1|X)")
    
    # Decision boundary line (P = 0.5)
    axes[0].contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")
    
    # Scatter plot data points
    axes[0].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
                    color="#3b82f6", edgecolors="k", label="Class 0", alpha=0.9)
    axes[0].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
                    color="#ef4444", edgecolors="k", label="Class 1", alpha=0.9)
    
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Feature X1")
    axes[0].set_ylabel("Feature X2")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Loss Curve Plot
    axes[1].plot(model.cost_history, color="#8b5cf6", linewidth=2)
    axes[1].set_title("Binary Cross-Entropy Loss Decay", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch Iterations")
    axes[1].set_ylabel("Log Loss")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig


def plot_loss_contours(X, y, model, w_range=(-2, 8), b_range=(-2, 8)):
    """
    Plot 2D loss landscape contour plot for Linear Regression showing gradient descent path.
    """
    w_vals = np.linspace(w_range[0], w_range[1], 100)
    b_vals = np.linspace(b_range[0], b_range[1], 100)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.zeros_like(W)

    m = X.shape[0]
    X_flat = X.flatten()
    y_flat = y.flatten()

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_ij = W[i, j]
            b_ij = B[i, j]
            y_pred = X_flat * w_ij + b_ij
            Z[i, j] = (1 / (2 * m)) * np.sum((y_pred - y_flat) ** 2)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contour(W, B, Z, levels=30, cmap="viridis")
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot weight history trajectory if available
    if len(model.weight_history) > 0:
        w_hist = [w.item() for w in model.weight_history]
        b_hist = [model.bias for _ in range(len(w_hist))] # simple tracking
        ax.plot(w_hist, b_hist, 'ro-', markersize=3, label="Gradient Descent Trajectory")
        ax.scatter(w_hist[-1], b_hist[-1], color="yellow", s=100, zorder=5, edgecolors="k", label="Final Convergence")

    ax.set_title("Loss Surface J(w, b) Contour Plot", fontsize=12, fontweight="bold")
    ax.set_xlabel("Weight (w)")
    ax.set_ylabel("Bias (b)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
