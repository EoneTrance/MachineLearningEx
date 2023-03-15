import numpy as np

loaded_data = np.loadtxt('./data-01.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[:, 0:-1]
t_data = loaded_data[:, [-1]]

print("x_data.ndim: ", x_data.ndim)
print("x_data.shape: ", x_data.shape)
print("t_data.ndim: ", t_data.ndim)
print("t_data.shape: ", t_data.shape)
print(x_data)
print(t_data)