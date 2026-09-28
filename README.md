# Machine Learning from Scratch: Linear & Logistic Regression

A comprehensive, zero-dependency (pure `NumPy`) implementation of **Linear Regression** and **Logistic Regression** built step-by-step from mathematical first principles.

Includes vectorized batch & mini-batch gradient descent, closed-form analytical normal equation, L1/L2 regularization, evaluation metrics suite, matplotlib visualization scripts, and a real-time **Interactive Web Dashboard**.

---

## 📐 Mathematical Derivations & Formulas

### 1. Linear Regression

#### **Hypothesis & Prediction**
Given input matrix $X \in \mathbb{R}^{m \times n}$ ($m$ samples, $n$ features) and target vector $y \in \mathbb{R}^{m \times 1}$:

$$\hat{y} = h_\theta(X) = X w + b$$

where $w \in \mathbb{R}^{n \times 1}$ are weights and $b \in \mathbb{R}$ is the bias.

#### **Cost Function (Mean Squared Error + L1/L2 Regularization)**
$$J(w, b) = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2 + \frac{\lambda_2}{2m} \|w\|_2^2 + \frac{\lambda_1}{m} \|w\|_1$$

#### **Gradients (Vectorized)**
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda_2}{m} w + \frac{\lambda_1}{m} \text{sign}(w)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})$$

#### **Closed-Form Solution (Normal Equation with L2 Ridge)**
Setting $\nabla J = 0$ yields the analytical optimal solution:

$$w_{\text{optimal}} = \left( X_b^T X_b + \lambda_2 I \right)^{-1} X_b^T y$$

where $X_b = [1 \quad X]$ includes the bias intercept column.

---

### 2. Logistic Regression

#### **Hypothesis & Sigmoid Activation**
For binary classification $y \in \{0, 1\}$:

$$z = X w + b$$

$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

#### **Cost Function (Binary Cross-Entropy / Log Loss)**
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right] + \frac{\lambda_2}{2m} \|w\|_2^2$$

#### **Gradients (Vectorized)**
Remarkably, the derivative of Binary Cross-Entropy with Sigmoid simplifies to the exact same algebraic form as Linear Regression:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (a - y) + \frac{\lambda_2}{m} w$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)} - y^{(i)})$$

#### **Decision Boundary**
Predict $\hat{y} = 1$ if $a = \sigma(X w + b) \ge \text{threshold}$ (default $0.5$), which corresponds to the linear hyperplane decision boundary:

$$w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = 0$$

---

## 📂 Project Structure

```
ml_from_scratch/
├── src/
│   ├── __init__.py               # Package exporter
│   ├── utils.py                   # Data generators, scalers, train_test_split, evaluation metrics
│   ├── linear_regression.py       # Linear Regression (GD + Normal Eq + L1/L2)
│   └── logistic_regression.py      # Logistic Regression (Sigmoid + LogLoss + Decision Boundary)
├── main.py                        # Verification & demonstration execution script
├── visualize.py                   # Matplotlib plotting functions (Fitted lines, contours, loss curves)
├── index.html                     # Interactive Web Visualizer Dashboard
├── app.js                         # Real-time Web Canvas ML Engine (JavaScript)
├── styles.css                     # Glassmorphism UI styling
├── linear_regression_results.png  # Generated matplotlib output chart
└── logistic_regression_results.png# Generated matplotlib output chart
```

---

## 🚀 Quick Start Guide

### 1. Run Python Demonstrations
Execute the python benchmark script to train both models and output evaluation metrics & plots:

```bash
python main.py
```

### 2. Launch Interactive Web Visualizer
Open `index.html` directly in your browser or run a lightweight Python HTTP server:

```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000` in your web browser.

---

## 🧪 Evaluation Metrics Included

- **Linear Regression**: Mean Squared Error ($MSE$), Root MSE ($RMSE$), Mean Absolute Error ($MAE$), Coefficient of Determination ($R^2$).
- **Logistic Regression**: Accuracy, Precision, Recall, $F_1$ Score, Confusion Matrix ($TN, FP, FN, TP$).
