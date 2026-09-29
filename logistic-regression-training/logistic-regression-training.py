import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    m, n = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(n)
    b = 0.0
    
    for _ in range(steps):
        # 1. Forward pass: compute linear combination and sigmoid activations
        z = np.dot(X, w) + b
        y_pred = 1 / (1 + np.exp(-z))
        
        # 2. Compute gradients (partial derivatives of binary cross-entropy loss)
        error = y_pred - y
        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)
        
        # 3. Backward pass: update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b
    # Write code here
    pass