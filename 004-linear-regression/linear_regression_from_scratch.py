"""
LINEAR REGRESSION FROM SCRATCH
==============================
Your first ML model. No scikit-learn, no PyTorch — just NumPy.

This script teaches you the entire ML training loop:
  1. Forward pass:  make a prediction using weights (matrix math)
  2. Loss:          measure how wrong the prediction is
  3. Gradient:      compute which direction to adjust weights
  4. Update:        nudge weights in the direction that reduces loss
  5. Repeat

THE PROBLEM:
We'll predict house prices from two features: square footage and number of bedrooms.
This is fake data, but the process is identical to real ML.
"""

import numpy as np

np.random.seed(42)  # for reproducibility

# =============================================================================
# STEP 1: Create some fake training data
# =============================================================================
# Imagine houses where price ≈ $200/sqft + $10,000/bedroom + $50,000 base
# We add noise because real data is never perfect.

num_samples = 100

sqft = np.random.uniform(800, 3000, num_samples)       # square footage
bedrooms = np.random.randint(1, 6, num_samples).astype(float)  # 1-5 bedrooms

# True relationship (the model doesn't know this — it has to discover it)
true_price = 200 * sqft + 10000 * bedrooms + 50000
noise = np.random.normal(0, 20000, num_samples) # real-world noise
price = true_price + noise

# Stack features into a matrix: each row is a house, each column is a feature
# Shape: (100, 2) — 100 houses, 2 features
X = np.column_stack([sqft, bedrooms])
y = price  # shape: (100,)

print(f"Training data: {X.shape[0]} houses, {X.shape[1]} features")
print(f"First house:   {X[0][0]:.0f} sqft, {X[0][1]:.0f} bedrooms, ${y[0]:,.0f}")
print()

# =============================================================================
# STEP 2: Normalize features (IMPORTANT!)
# =============================================================================
# Square footage ranges 800-3000, bedrooms range 1-5.
# Without normalization, gradient descent will zigzag inefficiently because
# the gradients for sqft will be tiny and for bedrooms will be huge.
#
# We scale each feature to have mean=0 and std=1.
# This is called "standardization" and you'll do it in almost every ML project.

X_mean = X.mean(axis=0)   # mean of each feature
X_std = X.std(axis=0)     # std of each feature
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

print(f"After normalization:")
print(f"  Feature means: {X_norm.mean(axis=0)} (should be ~0)")
print(f"  Feature stds:  {X_norm.std(axis=0)}  (should be ~1)")
print()

# =============================================================================
# STEP 3: Initialize the model
# =============================================================================
# Linear regression: prediction = X @ w + b
#   w = weights vector (one weight per feature)
#   b = bias (a single number, like the y-intercept)
#
# We start with random small values. The model will learn the correct values.

w = np.random.randn(2) * 0.01  # 2 weights (one per feature)
b = 0.0                         # bias

print(f"Initial weights: {w}")
print(f"Initial bias:    {b}")
print()

# =============================================================================
# STEP 4: The training loop — THIS IS THE CORE OF ALL ML
# =============================================================================

learning_rate = 0.1
num_epochs = 200      # how many times we loop through the data

print("Training...")
print(f"{'Epoch':>6} {'Loss':>12}")
print("-" * 20)

for epoch in range(num_epochs):

    # --- FORWARD PASS ---
    # Make predictions using current weights.
    # This is a matrix-vector multiplication (dot product for each sample).
    # X_norm shape: (100, 2), w shape: (2,) → predictions shape: (100,)
    predictions = X_norm @ w + b

    # --- COMPUTE LOSS ---
    # Mean Squared Error: average of (prediction - actual)²
    # This single number tells us "how wrong is our model overall?"
    errors = predictions - y_norm
    loss = np.mean(errors ** 2)

    # --- COMPUTE GRADIENTS ---
    # These tell us: "if I nudge each weight a tiny bit,
    # how much does the loss change?"
    #
    # For MSE with linear regression, the math works out to:
    #   dL/dw = (2/N) * X^T @ errors
    #   dL/db = (2/N) * sum(errors)
    #
    # Don't worry about deriving this yet — just note that it's
    # the chain rule applied to matrix operations.
    n = len(y_norm)
    grad_w = (2 / n) * X_norm.T @ errors   # gradient for weights
    grad_b = (2 / n) * np.sum(errors)       # gradient for bias

    # --- UPDATE WEIGHTS ---
    # Step in the OPPOSITE direction of the gradient (downhill).
    # learning_rate controls how big each step is.
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    # Print progress every 20 epochs
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"{epoch:>6} {loss:>12.6f}")

print()

# =============================================================================
# STEP 5: Examine what the model learned
# =============================================================================
# The learned weights are in normalized space. Let's convert back to
# the original scale to see if the model found the true relationship.

# Convert normalized weights back to original scale
w_original = w * (y_std / X_std)
b_original = b * y_std + y_mean - np.sum(w_original * X_mean)

print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Learned:  price = {w_original[0]:.1f} * sqft + {w_original[1]:.1f} * bedrooms + {b_original:.1f}")
print(f"Actual:   price = 200.0 * sqft + 10000.0 * bedrooms + 50000.0")
print()

# =============================================================================
# STEP 6: Make a prediction on a new house
# =============================================================================
new_house = np.array([2000, 3])  # 2000 sqft, 3 bedrooms
predicted_price = new_house @ w_original + b_original
actual_price = 200 * 2000 + 10000 * 3 + 50000

print(f"New house: 2000 sqft, 3 bedrooms")
print(f"  Predicted price: ${predicted_price:,.0f}")
print(f"  True price:      ${actual_price:,.0f}")
print()

# =============================================================================
# EXERCISES FOR YOU
# =============================================================================
# 1. Try changing the learning_rate to 0.01 and 1.0. What happens?
#    (Too slow convergence vs. overshooting)
#
# 2. Try removing the normalization step. Does it still converge?
#    (Hint: you may need a much smaller learning rate)
#
# 3. Add a third feature (e.g., "age of house") to X and update the
#    true price formula. Does the model learn three weights correctly?
#
# 4. Change num_samples to 10. Does the model still learn well?
#    (This illustrates why more data helps)
#
# 5. CHALLENGE: Track the loss at each epoch and plot it with matplotlib.
#    You should see a curve that drops sharply then flattens — this is
#    the "loss curve" and you'll see it in every ML project.
