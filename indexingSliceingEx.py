import numpy as np

# vector 생성
A = np.array([10, 20, 30, 40, 50, 60]).reshape(3, 2) # 3 X 2 행렬

print("A.shape: ", A.shape)
print(A)
print("A[0, 0]: ", A[0, 0])
print("A[0][0]: ", A[0][0])
print("A[2, 1]: ", A[2, 1])
print("A[2][1]: ", A[2][1])
print("A[0:-1, 1:2]: ", A[0:-1, 1:2]) # 0행부터 -1행 이전 인덱스까지, 1열부터 2열 이전 인덱스까지
print("A[:, 0]: ", A[:, 0])
print("A[:, :]: ", A[:, :])