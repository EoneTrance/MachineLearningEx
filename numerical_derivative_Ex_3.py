import numpy as np


# f 는 다변수 함수, x 는 모든 변수를 포함 하고 있는 numpy 객체 (배열, 행렬...)
def numerical_derivative(f, x):
    delta_x = 1e-4
    
    # 계산된 수치미분 값 저장 변수를 x 크기와 동일한 크기의 객체를 0으로 초기화 한 객체
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    # 변수의 개수 만큼 반복
    while not it.finished:
        idx = it.multi_index

        # numpy 타입은 mutable 이므로 원래 값 보관
        tmp_val = x[idx]
        
        #하나의 변수에 대한 수치미분 계산
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x + delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x - delta_x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


def my_func1(x):
    return 3 * x * (np.exp(x))
