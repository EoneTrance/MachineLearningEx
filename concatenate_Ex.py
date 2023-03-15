import numpy as np

# vector 생성
A = np.array([ [10, 20, 30], [50, 60, 70] ]) # 2 X 4 행렬

print("A.shape: ", A.shape)

# A matrix에 행(row) 추가할 행렬/ 1행 3열로 reshape
# 행을 추가하기 때문에 우선 열을 3열로 만들어야 함.
row_add = np.array([70, 80, 90]).reshape(1, 3)
print("row_add.shape: ", row_add.shape)

# A matrix에 열(column) 추가할 행렬. 2행 1열로 생성
# 열을 추가하기 때문에 우선 행을 2행으로 만들어야 함.
column_add = np.array([1000, 2000]).reshape(2, 1)
print("column_add.shape: ", column_add.shape)

# numpy.concatenate 에서 axis = 0 행(row) 기준
# A 행렬에 row_add 행렬 추가
B = np.concatenate((A, row_add), axis=0)
print(B)

#numpy.concatenate 에서 axis = 1 열(column) 기준
# A 행렬에 column_add 행렬 추가
C = np.concatenate((A, column_add), axis=1)
print(C)