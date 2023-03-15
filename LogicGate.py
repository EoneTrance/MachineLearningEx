import numpy as np

class LogicGate:
    def __init__(self, gate_name, xdata, tdata): # xdata, tdata => numpy.array(...)
        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2) #  입력데이터는 (0,0), (0,1), (1,0), (1,1) 총 4가지
        self.__tdata = tdata.reshape(4, 1)

        # 가중치 W, 바이어스 b 초기화
        self.__W = np.random.rand(2, 1)
        self.__b = np.random.rand(1)

        # 학습률 learning_rate 초기화
        self.__learning_rate = 1e-2

    # 손실함수
    def __loss_func(self):
        delta = 1e-7 # log 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = self.sigmoid(z)

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))

    # 손실 값 계산
    def error_Val(self):
        delta = 1e-7 # log 무한대 발산 방지

        z = np.dot(self.__xdata, self.__W) + self.__b
        y = self.sigmoid(z)

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))
    
    # sigmoid 함수
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):
        f = lambda x : self.__loss_func()

        print("Initial erorr value", self.error_Val())

        for step in range(8001):
            self.__W -= self.__learning_rate * self.numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * self.numerical_derivative(f, self.__b)
            if step % 400 == 0:
                print("step: ", step, "error value: ", self.error_Val())

    # 수치미분 함수
    def numerical_derivative(self, f, x):
        delta_x = 1e-4
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]

            x[idx] = float(tmp_val) + delta_x
            fx1 = f(x)  # f(x+delta_x)

            x[idx] = tmp_val - delta_x
            fx2 = f(x)  # f(x-delta_x)

            grad[idx] = (fx1 - fx2) / (2 * delta_x)

            x[idx] = tmp_val

            it.iternext()

        return grad

    # 미래 값 예측 함수
    def predict(self, input_data):
        z = np.dot(input_data, self.__W) + self.__b
        y = self.sigmoid(z)

        if y > 0.5:
            result = 1 # True
        else:
            result = 0 # False

        return y, result
