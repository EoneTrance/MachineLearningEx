import numpy as np


class LogicGateDL:
    def __init__(self, gate_name, xdata, tdata): # xdata, tdata => numpy.array(...)
        self.name = gate_name

        # 입력 데이터, 정답 데이터 초기화
        self.__xdata = xdata.reshape(4, 2) #  4개의 입력데이터 x1, x2 에 대하여 batch 처리 행렬
        self.__tdata = tdata.reshape(4, 1) #  4개의 입력데이터 x1, x2 에 대한 각각의 계산 값 행렬

        # 2층 hidden layer unit : 6개 가정, 가중치 W2, 바이어스 b2 초기화
        self.__W2 = np.random.rand(2, 6) # weight, 2 X 6 matrix
        self.__b2 = np.random.rand(6)

        # 3층 hidden layer unit : 1개, 가중치 W3, 바이어스 b3 초기화
        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        # 학습률 learning_rate 초기화
        self.__learning_rate = 1e-2

        print(self.name + " object is crated")

    # feed forward를 통하여 손실함수(cross-entropy)값 계산
    def feed_forward(self):
        delta = 1e-7 # log 무한대 발산 방지

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2 # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2) # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3 # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3) # 출력층의 출력

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))


    # 외부 출력을 위한 손실함수(cross-entropy) 값 계산
    def loss_val(self):
        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.__xdata, self.__W2) + self.__b2  # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3  # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3)  # 출력층의 출력

        # cross-entropy
        return -np.sum(self.__tdata*np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta))

    # sigmoid 함수
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    # 수치미분을 이용하여 손실함수가 최소가 될때 까지 학습하는 함수
    def train(self):
        f = lambda x : self.feed_forward()

        print("Initial loss value: ", self.loss_val())

        for step in range(10001):
            self.__W2 -= self.__learning_rate * self.numerical_derivative(f, self.__W2)
            self.__b2 -= self.__learning_rate * self.numerical_derivative(f, self.__b2)
            self.__W3 -= self.__learning_rate * self.numerical_derivative(f, self.__W3)
            self.__b3 -= self.__learning_rate * self.numerical_derivative(f, self.__b3)

            if step % 400 == 0:
                print("step: ", step, " loss value: ", self.loss_val())

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

            grad[idx] = (fx1 - fx2) / (2*delta_x)

            x[idx] = tmp_val

            it.iternext()

        return grad

    # 미래 값 예측 함수
    def predict(self, data):
        z2 = np.dot(data, self.__W2) + self.__b2 # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2) # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3 # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3) # 출력층의 출력

        if y > 0.5:
            result = 1 # True
        else:
            result = 0 # False

        return y, result
