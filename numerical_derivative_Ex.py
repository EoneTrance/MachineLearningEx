def my_func1(x):
    return x**2


def numerical_derivative(f, x):
    delta_x = 1e-4
    return (f(x + delta_x) - f(x - delta_x)) / (2 * delta_x)


result = numerical_derivative(my_func1, 3)
print(result)
