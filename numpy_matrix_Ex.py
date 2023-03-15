import numpy as np

a = [ [1, 0], [0, 1] ]
b = [ [1, 1], [1, 1] ]
c = a + b
print(c)

d = np.array(a)
e = np.array(b)
f = d + e
print("d: ", d)
print("e: ", e)
print("f: ", f)

# vector 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# vector a, b 출력
print("a: ", a, ", b: ", b)

# vector a, b 형상 출력
print("a.shape: ", a.shape, ", b.shape: ", b.shape)

# vector a, b 차원 출력
print("a.ndim: ", a.ndim, ", b.ndim: ", b.ndim)

# vector 산술 연산
print("a + b: ", a + b)
print("a - b: ", a - b)
print("a * b: ", a * b)
print("a / b: ", a / b)

# vector 생성
c = np.array([ [1, 2, 3], [4, 5, 6]])
d = np.array([ [-1, -2, -3], [-4, -5, -6]])

# matrix a, b 형상 출력
print("c.shape: ", c.shape)
print("d.shape: ", d.shape)

# matrix a, b 차원 출력
print("c.ndim: ", c.ndim)
print("d.ndim: ", d.ndim)

# vector 생성
e = np.array([1, 2, 3])

# vector 형상 출력
print("e.shape: ", e.shape)
print(e)

# vector를 (1, 3) 행렬로 변환
e = e.reshape(1, 3)

print("e.shape: ", e.shape)
print(e)