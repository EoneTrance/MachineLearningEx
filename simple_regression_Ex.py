import numpy as np


x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
y_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

# raw_data = [ [1, 2], [2, 3], [3, 4], [4, 5], [5, 6] ]

W = np.random.rand(1, 1)
b = np.random.rand(1)
print("W: ", W, ", W.shape: ", W.shape)
print("b: ", b, ", b.shape: ", b.shape)


def loss_func(x, t):
    y = np.dot(x, W) + b

    return (np.sum((t - y)**2)) / (len(x))