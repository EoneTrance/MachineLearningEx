import numpy as np


class BackPropagation:
    def __init__(self, i_nodes, h_nodes, o_nodes, leaning_rate):
        self.i_nodes = i_nodes # input_nodes = 입력층 노드 갯수
        self.h_nodes = h_nodes # hidden_nodes = 은닉층 노드 갯수
        self.o_nodes = o_nodes # output_nodes = 출력층 노드 갯수

        # 은닉층 가중치 W2 = (784 X 100) Xavier /He 방법으로 self.W2 가중치 초기화
        self.W2 = np.random.randn(self.i_nodes, self.h_nodes) / np.sqrt((self.i_nodes/2))
        self.b2 = np.random.rand(self.h_nodes)

        # 출력층 가중치는 W3 = (100 X 10) Xavier /He 방법으로 self.W3 가중치 초기화
        self.W3 = np.random.randn(self.h_nodes, self.o_nodes) / np.sqrt((self.h_nodes/2))
        self.b3 = np.random.rand(self.o_nodes)

        # 출력층 선형회귀 값 Z3, 출력값 A3 정의 (모두 행렬로 표시)
        self.Z3 = np.zeros(([1, self.o_nodes]))
        self.A3 = np.zeros(([1, self.o_nodes]))

        # 은닉층 선형회귀 값 Z2, 출력값 A2 정의 (모두 행렬로 표시)
        self.Z2 = np.zeros(([1, self.h_nodes]))
        self.A2 = np.zeros(([1, self.h_nodes]))

        # 입력층 선형회귀 값 Z1, 출력값 A1 정의 (모두 행렬로 표시)
        self.Z1 = np.zeros(([1, self.i_nodes]))
        self.A1 = np.zeros(([1, self.i_nodes]))

        # 학습률 learning_rate 초기화
        self.learning_rate = leaning_rate

    # feed forward를 이용하여 입력층에서 부터 출력층까지 데이터를 전달하고 손실 함수 값 계산
    # loss_val(self) 메서드와 동일한 코드, loss_val(self)은 외부 출력용으로 사용됨
    def feed_forward(self):
        delta = 1e-7 # log 무한대 발산 방지

        # 입력층 선형회귀 값 Z1, 출력값 A1 계산
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2, 출력값 A2 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3, 출력값 A3 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        # cross-entropy
        return -np.sum(self.target_data*np.log(self.A3+delta) + (1-self.target_data)*np.log((1-self.A3)+delta))

    # 외부 출력을 위한 손실함수(cross-entropy) 값 계산
    def loss_val(self):
        delta = 1e-7  # log 무한대 발산 방지

        # 입력층 선형회귀 값 Z1, 출력값 A1 계산
        self.Z1 = self.input_data
        self.A1 = self.input_data

        # 은닉층 선형회귀 값 Z2, 출력값 A2 계산
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        # 출력층 선형회귀 값 Z3, 출력값 A3 계산
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        # cross-entropy
        return -np.sum(self.target_data*np.log(self.A3+delta) + (1-self.target_data)*np.log((1-self.A3)+delta))

    # sigmoid 함수
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

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

    # input_data : 784개, target_data : 10개
    def train(self, input_data, target_data):
        self.target_data = target_data
        self.input_data = input_data

        # 먼저 feed forward 를 통해서 최종 출력값과 이를 바탕으로 현재의 에러 값 계산
        loss_val = self.feed_forward()

        # 출력층 loss 인 loss_3 구함
        loss_3 = (self.A3-self.target_data) * self.A3 * (1-self.A3)

        # 출력층 가중치 W3, 출력층 바이어스 b3 업데이트
        self.W3 = self.W3 - self.learning_rate * np.dot(self.A2.T, loss_3)
        self.b3 = self.b3 - self.learning_rate * loss_3

        # 은닉층 loss 인 loss_2 구함
        loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1-self.A2)

        # 은닉층 가중치 W2, 은닉층 바이어스 b2 업데이트
        self.W2 = self.W2 - self.learning_rate * np.dot(self.A1.T, loss_2)
        self.b2 = self.b2 - self.learning_rate * loss_2

    # input_data 는 행렬로 입력됨 즉, (1, 784) shape 을 가짐
    def predict(self, input_data):
        Z2 = np.dot(input_data, self.W2) + self.b2 # 은닉층의 선형회귀 값
        A2 = self.sigmoid(Z2) # 은닉층의 출력

        Z3 = np.dot(A2, self.W3) + self.b3 # 출력층의 선형회귀 값
        A3 = self.sigmoid(Z3) # 출력층의 출력

        #가장 큰 값을 가지는 인덱스를 정답으로 인식함(argmax) 즉, one-hot encoding을 구현함
        predicted_num = np.argmax(A3)

        return predicted_num

    # 정확도 측정함수
    def accuracy(self, x_test_data, t_test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(x_test_data)):
            label = int(t_test_data[index])

            # one-hot encoding 을 위한 데이터 정규화 (data normalize)
            # 입력 데이터는 0~255 이기 떄문에 가끔 overflow 발생. 따라서 784개의 모든 입력 값을 0~1 사이의 값으로 normalize 함
            data = (x_test_data[index] / 255 * 0.99) + 0.01

            # predict 를 위해서 vector 를 matrix 로 변환하여 인수로 넘겨줌
            predicted_num = self.predict(np.array(data, ndmin=2))

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        print("Current Accuracy: ", 100*(len(matched_list) / (len(x_test_data))), "%")

        return matched_list, not_matched_list