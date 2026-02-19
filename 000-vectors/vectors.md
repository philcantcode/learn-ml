# Vectors

A vector from a developer perspective is an array of numbers:

```python
a = [1, 2, 3]
b = [4, 5, 6]
```

From a machine learning perspective a vector represents an `arrow in feature space`.

```pseudo
house_1 = [250000, 4]
house_2 = [400000, 6]
```

In the exaple above a and b represent [house_price, num_bedrooms]. Feature space is a theoretical space space containing all of the values - not an actual directional space. 

## Dot Product

The dot product tells you how similar two vectors.

```python
a = [1, 2, 3]
b = [4, 5, 6]
dot = sum(x * y for x, y in zip(a, b))  # 1*4 + 2*5 + 3*6 = 32
```

In python `zip(a, b)` pairs up corresponding elements from both lists.

zip(a, b) produces: (1, 4), (2, 5), (3, 6)

This tells you how similar two vectors are because it multiplies the values. Multiplication amplifies agreement:

- When both values being multiplied are large and positive, the product is large and positive (strong agreement);
- When both are small, the product is small (weak agreement);
- When signs differ, the product is negative (disagreement);

```python
a = [10, 10, 0]
b = [10, 10, 0]
dot = 10*10 + 10*10 + 0*0 = 200  # very similar

a = [10, 10, 0]
c = [0, 0, 10]
dot = 10*0 + 10*0 + 0*10 = 0     # orthogonal, no similarity

a = [10, 10, 0]
d = [-10, -10, 0]
dot = 10*(-10) + 10*(-10) + 0*0 = -200  # opposite direction
```

