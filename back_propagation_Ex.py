import numpy as np
from datetime import datetime
from keras.datasets import mnist
from BackPropagation import BackPropagation

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epochs = 1

nn = BackPropagation(input_nodes, hidden_nodes, output_nodes, learning_rate)
(x_train_data, t_train_data), (x_test_data, t_test_data) = mnist.load_data()

x_train_data = x_train_data.reshape(60000, 784)
x_test_data = x_test_data.reshape(10000, 784)

start_time = datetime.now()

for i in range(epochs):
    for step in range(len(x_train_data)):  # training

        # input_Data, target_data normalize
        target_data = np.zeros((output_nodes)) + 0.01
        target_data[int(t_train_data[step])] = 0.99
        input_data = ((x_train_data[step] / 255.0) * 0.99) + 0.01

        nn.train(np.array(input_data, ndmin=2), np.array(target_data, ndmin=2))

        if step % 400 == 0:
            print("step: ", step, "loss_val: ", nn.loss_val())

end_time = datetime.now()
print("Elapsed time: ", end_time - start_time)

nn.accuracy(x_test_data, t_test_data)