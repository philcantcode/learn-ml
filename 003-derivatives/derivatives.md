# Derivatives

A derivative tells you how a function's output changes when you nudge its input.

```python
# A simple function
def f(x):
    return x ** 2

# Its derivative: tells us the slope at any point
def df(x):
    return 2 * x
```

At x = 3, the derivative is 6 — meaning if you increase x a tiny bit, f(x) increases by about 6 times that tiny bit. At x = -2, the derivative is -4 — meaning increasing x would decrease the output.

Now extend this to multiple variables. If your function takes a vector as input (which every ML model does), the gradient is just the vector of all partial derivatives — one per input dimension. It points in the direction of steepest increase.

```python
# f(x, y) = x² + 3y²
# gradient = [df/dx, df/dy] = [2x, 6y]

def gradient(x, y):
    return np.array([2*x, 6*y])

# At point (1, 2): gradient is [2, 12]
# This points "uphill" — toward where f increases fastest
```

Gradient descent — the algorithm that trains nearly every ML model — simply says: go the opposite direction of the gradient to make the function smaller. If the gradient points uphill, negate it to go downhill. The function you're minimizing is the loss function — a measure of how wrong the model currently is.

```python
# Gradient descent in 5 lines
x = 5.0               # start somewhere
learning_rate = 0.1
for _ in range(50):
    grad = 2 * x      # derivative of x²
    x = x - learning_rate * grad  # step opposite to gradient
# x converges toward 0 — the minimum of x²
```