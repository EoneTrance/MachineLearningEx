import numpy as np

X = np.array([2, 3, 4, 5, 6, 7, 8])

print("np.max(X): ", np.max(X))
print("np.min(X): ", np.min(X))
print("np.argmax(X): ", np.argmax(X))
print("np.argmin(X): ", np.argmin(X))

X = np.array([ [2, 4, 6], [1, 2, 3], [0, 5, 8] ])

print("np.max(X): ", np.max(X, axis=0)) # axis=0 열기준
print("np.min(X): ", np.min(X, axis=0)) # axis=0 열기준

print("np.max(X): ", np.max(X, axis=1)) # axis=1 행기준
print("np.min(X): ", np.min(X, axis=1)) # axis=1 행기준

print("np.argmax(X): ", np.argmax(X, axis=0)) # axis=0 열기준
print("np.argmin(X): ", np.argmin(X, axis=0)) # axis=0 열기준

print("np.argmax(X): ", np.argmax(X, axis=1)) # axis=1 행기준
print("np.argmin(X): ", np.argmin(X, axis=1)) # axis=1 행기준

A = np.ones([3, 3])

print("A.shape: ", A.shape)
print(A)

B = np.zeros([3, 2])
print("B.shape: ", B.shape)
print(B)