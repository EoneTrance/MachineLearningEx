import numpy as np

# vector 생성
A = np.array([ [10, 20, 30, 40], [50, 60, 70, 80] ]) # 2 X 4 행렬

print(A, "\n")
print("A.shape: ", A.shape, "\n")

# 행렬 A의 iterator 생성

it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:
    idx = it.multi_index
    print("current value: ", A[idx])
    it.iternext()
    