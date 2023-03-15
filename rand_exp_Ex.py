import numpy as np

random_number1 = np.random.rand(3)
random_number2 = np.random.rand(10, 3)
random_number3 = np.random.rand(3, 1)

print("random_number1: ", random_number1)
print("random_number1.shape: ", random_number1.shape)
print("random_number1: ", random_number2)
print("random_number1.shape: ", random_number2.shape)
print("random_number1: ", random_number3)
print("random_number1.shape: ", random_number3.shape)

X = np.array([2, 3, 4, 5, 6, 7, 8])
print("np.sum(X): ", np.sum(X))
print("np.exp(X): ", np.exp(X))
print("np.log(X): ", np.log(X))
print(np.exp(2))
print(2 * 0.69314718)