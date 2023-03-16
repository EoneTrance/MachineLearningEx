import numpy as np
from keras.datasets import mnist
from NeuralNetwork import NeuralNetwork

input_nodes = 28
hidden_nodes = 100
output_nodes = 10

nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes)
(x_train_data, t_train_data), (x_test_data, t_test_data) = mnist.load_data()

for step in range(30001): # 전체 training data의 50%
    # 총 60,000 개의 training data 가운데 random하게 30,000개 선택
    index = np.random.randint(0, len(x_train_data) - 1)
    nn.train(x_train_data[index], t_train_data[index])

    if step % 400 == 0:
        print("step: ", step, "loss_val: ", nn.loss_val())

nn.accuracy(x_test_data, t_test_data)