import numpy as np

def my_func1(x):
    return 3 * x * (np.exp(x))


def numerical_derivative(f, x):
    delta_x = 1e-4
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)


result = numerical_derivative(my_func1, 2)
print(result)
print("3 * exp(2) + 3 * 2 * 2xp(2): ", 3 * np.exp(2) + 3 * 2 * np.exp(2))