import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes # input_nodes = 입력층 노드 갯수
        self.hidden_nodes = hidden_nodes # hidden_nodes = 은닉층 노드 갯수
        self.output_nodes = output_nodes # output_nodes = 출력층 노드 갯수

        # 2층 hidden layer unit
        # 가중치 W, 바이어스 b 초기화
        self.W2 = np.random.rand(self.input_nodes, self.input_nodes, self.hidden_nodes)
        self.b2 = np.random.rand(self.hidden_nodes)

        # 3층 output later unit
        self.W3 = np.random.rand(self.hidden_nodes, self.output_nodes)
        self.b3 = np.random.rand(self.output_nodes)

        # 학습률 learning_rate 초기화
        self.__learning_rate = 1e-4

    # feed forward를 이용하여 입력층에서 부터 출력층까지 데이터를 전달하고 손실 함수 값 계산
    # loss_val(self) 메서드와 동일한 코드, loss_val(self)은 외부 출력용으로 사용됨
    def feed_forward(self):
        delta = 1e-7 # log 무한대 발산 방지

        z2 = np.dot(self.input_data, self.W2) + self.b2 # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2) # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3 # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3) # 출력층의 출력

        # cross-entropy
        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    # 외부 출력을 위한 손실함수(cross-entropy) 값 계산
    def loss_val(self):
        delta = 1e-7  # log 무한대 발산 방지

        z2 = np.dot(self.input_data, self.W2) + self.b2  # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2)  # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3  # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3)  # 출력층의 출력

        # cross-entropy
        return -np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    # sigmoid 함수
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    # input_data : 784개, target_data : 10개
    def train(self, x_training_data, t_training_data):
        #normalize
        #one-hot encoding을 위한 10개의 노드 0.01 초기화 및 정답을 나타내는 인덱스에 가장 큰 값인 0.99로 초기화
        self.target_data = np.zeros(self.output_nodes) + 0.01
        self.target_data[int(t_training_data)] = 0.99

        # 입력 데이터는 0~255 이기 때문에, 가끔 overflow 발생. 따라서 모든 입력 값을 0~1 사이의 값으로 normalize 함
        self.input_data = (x_training_data / 255.0 * 0.99) + 0.01

        f = lambda x : self.feed_forward()

        self.W2 -= self.__learning_rate * self.numerical_derivative(f, self.W2)
        self.b2 -= self.__learning_rate * self.numerical_derivative(f, self.b2)
        self.W3 -= self.__learning_rate * self.numerical_derivative(f, self.W3)
        self.b3 -= self.__learning_rate * self.numerical_derivative(f, self.b3)

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

    # query, 즉 미래 값 예측 함수
    def predict(self, data):
        z2 = np.dot(data, self.W2) + self.b2 # 은닉층의 선형회귀 값
        a2 = self.sigmoid(z2) # 은닉층의 출력

        z3 = np.dot(a2, self.W3) + self.b3 # 출력층의 선형회귀 값
        y = a3 = self.sigmoid(z3) # 출력층의 출력

        #가장 큰 값을 가지는 인덱스를 정답으로 인식함(argmax) 즉, one-hot encoding을 구현함
        predicted_num = np.argmax(y)

        return predicted_num

    def accuracy(self, x_test_data, t_test_data):
        matched_list = []
        not_matched_list = []

        for index in range(len(x_test_data)):
            label = int(t_test_data[index])
            data = x_test_data / 255 * 0.99 + 0.01
            predicted_num = self.predict(data)

            if label == predicted_num:
                matched_list.append(index)
            else:
                not_matched_list.append(index)

        print("Current Accuracy: ", 100*(len(matched_list) / (len(x_test_data))), "%")

        return matched_list, not_matched_list