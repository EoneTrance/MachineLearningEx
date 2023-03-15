import numpy as np

# vector 생성
a = np.array([ [1, 2, 3], [4, 5, 6] ]) # 2 X 3 행렬
b = np.array([ [-1, -2], [-3, -4], [-5, -6] ]) # 3 X 2 행렬

# 2 X 3 dot product 3 X 2 == 2 X 2 행렬
c = np.dot(a, b)

# matrix a, b 형상 출력
print("a.shape: ", a.shape)
print("b.shape: ", b.shape)
print("c.shape: ", c.shape)
print(a)
print(b)
print(c)