"""
NEURAL NETWORK FROM SCRATCH
============================
We build a neural network that learns a NONLINEAR pattern.
No PyTorch, no TensorFlow — just NumPy.

THE TASK:
Classify points inside vs outside a circle.
A linear model CANNOT learn this (it can only draw straight lines).
A neural network CAN — thanks to activation functions.

This builds directly on your linear regression code:
  - Same training loop (forward → loss → gradient → update)
  - Same matrix multiplication for each layer
  - NEW: ReLU activation between layers
  - NEW: backpropagation through multiple layers (chain rule)
"""

import numpy as np
np.random.seed(42)

# =============================================================================
# STEP 1: Generate nonlinear data — points inside/outside a circle
# =============================================================================
# This is the key: no single straight line can separate inside from outside.
# That's why we need a neural network.

num_samples = 300

# Random 2D points in [-2, 2] × [-2, 2]
X = np.random.uniform(-2, 2, (num_samples, 2))

# Label: 1 if inside circle of radius 1.2, else 0
distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
y = (distances < 1.2).astype(float).reshape(-1, 1)  # shape: (300, 1)

# Shuffle and split into train/test
indices = np.random.permutation(num_samples)
split = int(0.8 * num_samples)
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

print(f"Training: {X_train.shape[0]} samples")
print(f"Testing:  {X_test.shape[0]} samples")
print(f"Inside circle: {y_train.sum():.0f}/{len(y_train)}")
print()

# =============================================================================
# STEP 2: Activation functions
# =============================================================================
# ReLU for hidden layers — the nonlinearity that makes everything work
# Sigmoid for output — squashes to [0, 1] for binary classification

def relu(z):
    """ReLU: if positive, pass through. If negative, output 0."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: 1 if positive, 0 if negative.
    This is needed for backpropagation."""
    return (z > 0).astype(float)

def sigmoid(z):
    """Sigmoid: squash any number to range [0, 1].
    Used for binary classification — output = probability of class 1."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# =============================================================================
# STEP 3: Initialize the network
# =============================================================================
# Architecture: 2 inputs → 16 hidden neurons → 8 hidden neurons → 1 output
#
# Why these sizes? The hidden layers need enough neurons to carve up the
# 2D space into regions. 16 and 8 is plenty for a circle.
#
# Weight initialization matters! We use "He initialization" — scale by
# sqrt(2/n_inputs). Too large → gradients explode. Too small → gradients vanish.

def he_init(rows, cols):
    """He initialization: good for ReLU networks."""
    return np.random.randn(rows, cols) * np.sqrt(2.0 / cols)

# Layer 1: 2 inputs → 16 hidden
W1 = he_init(16, 2)    # shape: (16, 2)
b1 = np.zeros((16, 1))  # shape: (16, 1)

# Layer 2: 16 → 8 hidden
W2 = he_init(8, 16)     # shape: (8, 16)
b2 = np.zeros((8, 1))

# Layer 3 (output): 8 → 1
W3 = he_init(1, 8)      # shape: (1, 8)
b3 = np.zeros((1, 1))

print(f"Network architecture: 2 → 16 → 8 → 1")
print(f"Total parameters: {W1.size + b1.size + W2.size + b2.size + W3.size + b3.size}")
print()

# =============================================================================
# STEP 4: The training loop — now with BACKPROPAGATION
# =============================================================================
# The forward pass is: input → [W1 → ReLU] → [W2 → ReLU] → [W3 → Sigmoid] → prediction
# The backward pass computes gradients in REVERSE order using the chain rule.

learning_rate = 0.05
num_epochs = 500
batch_size = 32  # mini-batch gradient descent: update on small chunks, not all data

print("Training...")
print(f"{'Epoch':>6} {'Loss':>10} {'Train Acc':>12} {'Test Acc':>10}")
print("-" * 42)

for epoch in range(num_epochs):

    # Shuffle training data each epoch (helps optimization)
    perm = np.random.permutation(len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0

    # Process mini-batches
    for i in range(0, len(X_train), batch_size):
        Xb = X_shuffled[i:i+batch_size].T  # shape: (2, batch_size)
        yb = y_shuffled[i:i+batch_size].T  # shape: (1, batch_size)
        m = Xb.shape[1]  # actual batch size (might be smaller at end)

        # ============== FORWARD PASS ==============
        # Each layer: linear transform → activation
        # We save intermediate values — we'll need them for backprop.

        z1 = W1 @ Xb + b1       # linear transform, shape: (16, m)
        a1 = relu(z1)            # activation, shape: (16, m)

        z2 = W2 @ a1 + b2       # linear transform, shape: (8, m)
        a2 = relu(z2)            # activation, shape: (8, m)

        z3 = W3 @ a2 + b3       # linear transform, shape: (1, m)
        a3 = sigmoid(z3)         # final prediction [0, 1], shape: (1, m)

        # ============== COMPUTE LOSS ==============
        # Binary cross-entropy: the standard loss for classification.
        # It heavily penalizes confident wrong predictions.
        eps = 1e-8  # avoid log(0)
        loss = -np.mean(yb * np.log(a3 + eps) + (1 - yb) * np.log(1 - a3 + eps))
        epoch_loss += loss * m

        # ============== BACKWARD PASS (BACKPROPAGATION) ==============
        # This is the chain rule in action. We work backwards from the loss.
        #
        # Key idea: at each layer, we compute:
        #   dz = how much the loss changes w.r.t. this layer's pre-activation
        #   dW = how much the loss changes w.r.t. this layer's weights
        #   db = how much the loss changes w.r.t. this layer's biases
        #
        # Then we pass the gradient back to the previous layer.

        # Output layer gradient
        dz3 = a3 - yb                          # shape: (1, m)
        dW3 = (1/m) * dz3 @ a2.T               # shape: (1, 8)
        db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)  # shape: (1, 1)

        # Hidden layer 2 gradient — HERE'S THE CHAIN RULE
        # We take the gradient from layer 3 and push it back through layer 2
        da2 = W3.T @ dz3                        # pass gradient back, shape: (8, m)
        dz2 = da2 * relu_derivative(z2)          # multiply by ReLU derivative, shape: (8, m)
        dW2 = (1/m) * dz2 @ a1.T               # shape: (8, 16)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)  # shape: (8, 1)

        # Hidden layer 1 gradient — chain rule again
        da1 = W2.T @ dz2                        # shape: (16, m)
        dz1 = da1 * relu_derivative(z1)          # shape: (16, m)
        dW1 = (1/m) * dz1 @ Xb.T               # shape: (16, 2)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)  # shape: (16, 1)

        # ============== UPDATE WEIGHTS ==============
        # Same as before: step opposite the gradient
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    # Evaluate accuracy
    epoch_loss /= len(X_train)

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        # Forward pass on train and test sets
        def predict(X_data):
            a = relu(W1 @ X_data.T + b1)
            a = relu(W2 @ a + b2)
            a = sigmoid(W3 @ a + b3)
            return (a > 0.5).astype(float).T

        train_acc = np.mean(predict(X_train) == y_train)
        test_acc = np.mean(predict(X_test) == y_test)
        print(f"{epoch:>6} {epoch_loss:>10.4f} {train_acc:>11.1%} {test_acc:>10.1%}")

print()

# =============================================================================
# STEP 5: Compare with a LINEAR model on the same data
# =============================================================================
# To prove the point: a linear model CANNOT learn a circle.

print("=" * 42)
print("COMPARISON: Linear model on the same data")
print("=" * 42)

W_lin = np.random.randn(1, 2) * 0.01
b_lin = np.zeros((1, 1))

for epoch in range(500):
    z = W_lin @ X_train.T + b_lin
    a = sigmoid(z)
    m = X_train.shape[0]
    dz = a - y_train.T
    W_lin -= 0.05 * (1/m) * dz @ X_train
    b_lin -= 0.05 * (1/m) * np.sum(dz)

# Evaluate linear model
lin_pred = (sigmoid(W_lin @ X_test.T + b_lin) > 0.5).astype(float).T
lin_acc = np.mean(lin_pred == y_test)
print(f"Linear model test accuracy: {lin_acc:.1%}")
print(f"Neural network test accuracy: {test_acc:.1%}")
print()

if lin_acc < test_acc:
    ratio = (test_acc - lin_acc) / lin_acc * 100
    print("The neural network crushes the linear model because")
    print("a circle CANNOT be separated by a straight line.")
    print("The activation functions (ReLU) give the network the")
    print("ability to learn curved decision boundaries.")
else:
    print("(Unexpected — try running again with a different seed)")

print()

# =============================================================================
# STEP 6: Visualize the decision boundary
# =============================================================================
# Create a grid of points and see what the network predicts at each one.
# This shows you the SHAPE the network has learned.

print("Decision boundary (text visualization):")
print("  · = network predicts OUTSIDE")
print("  █ = network predicts INSIDE")
print("  Circle outline shown for reference")
print()

grid_size = 30
for row in range(grid_size):
    line = ""
    for col in range(grid_size):
        px = -2 + 4 * col / (grid_size - 1)
        py = -2 + 4 * row / (grid_size - 1)

        # Network prediction
        inp = np.array([[px, py]])
        a = relu(W1 @ inp.T + b1)
        a = relu(W2 @ a + b2)
        pred = sigmoid(W3 @ a + b3)[0, 0]

        # True boundary (circle)
        dist = np.sqrt(px**2 + py**2)
        on_boundary = abs(dist - 1.2) < 0.15

        if on_boundary:
            line += "○ "
        elif pred > 0.5:
            line += "█ "
        else:
            line += "· "
    print(line)

print()
print("The network learned a circular boundary — something")
print("impossible for a linear model with any amount of training.")

# =============================================================================
# EXERCISES
# =============================================================================
# 1. Remove the ReLU activations (replace relu(z) with just z).
#    Does the network still learn the circle? Why not?
#
# 2. Reduce the hidden layers to just 2 neurons each.
#    Can it still learn? What about 4 neurons?
#    (This is the "capacity" of the network)
#
# 3. Change the data to an XOR pattern:
#    y = 1 if (x1 > 0 and x2 > 0) or (x1 < 0 and x2 < 0)
#    This is the classic problem that killed 1960s AI (perceptrons couldn't do it).
#    Your neural network should handle it easily.
#
# 4. Try replacing ReLU with sigmoid in the hidden layers.
#    Does it still learn? Is it faster or slower?
#    (This demonstrates the "vanishing gradient problem")
#
# 5. CHALLENGE: Add a learning rate schedule — start at 0.05 and
#    decay by 0.95 every 50 epochs. Does it converge better?
