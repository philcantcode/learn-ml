# Matrices

A matrix can be seen as a function that transforms vectors.

```python
import numpy as np

# A 2x3 matrix: takes a 3D vector, outputs a 2D vector
W = np.array([[1, 2, 3],
              [4, 5, 6]])

x = np.array([1, 0, 1])

# In python the @ symbol is used for matrix multiplication
output = W @ x  # [1*1+2*0+3*1, 4*1+5*0+6*1] = [4, 10]
```

This is literally what a neural network layer does. Each layer is a matrix multiplication followed by a nonlinear function. The matrix W contains the weights — the learned parameters of the model. When people say a model has "70 billion parameters," they mean the total number of values across all these matrices.

The key intuition: training a model = finding the right numbers to put in these matrices.

How do you find those weights?