import numpy as np

# vector 생성
A = np.array([ [1, 2], [3, 4], [5, 6] ]) # 3 X 2 행렬
B = A.T # A의 전치행렬, 2 X 3 행렬

print("A.shape: ", A.shape)
print("B.shape: ", B.shape)
print(A)
print(B)

C = np.array([1, 2, 3, 4, 5]) # vector, matrix 아님
D = C.T

E = C.reshape(1, 5) # 1 X 5 matrix
F = E.T # E 의 전치행렬, 5 X 1 matrix (scalar)

print("C.shape: ", C.shape)
print("D.shape: ", D.shape)
print("E.shape: ", E.shape)
print("F.shape: ", F.shape)
print(C)
print(D)
print(E)
print(F)