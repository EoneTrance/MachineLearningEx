import numpy as np
from LogicGateDL import LogicGateDL

x_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

# AND Gate ==================================================
t_data = np.array([0, 0, 0, 1])

AND_obj = LogicGateDL("AND", x_data, t_data)
AND_obj.train()

# prediction
print(AND_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = AND_obj.predict(data)
    print(sigmoid_val, " = ", logical_val)

# OR Gate ==================================================
t_data = np.array([0, 1, 1, 1])

OR_obj = LogicGateDL("OR", x_data, t_data)
OR_obj.train()

# prediction
print(OR_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = OR_obj.predict(data)
    print(sigmoid_val, " = ", logical_val)

# NAND Gate ==================================================
t_data = np.array([1, 1, 1, 0])

NAND_obj = LogicGateDL("NAND", x_data, t_data)
NAND_obj.train()

# prediction
print(NAND_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(data)
    print(sigmoid_val, " = ", logical_val)

# XOR Gate ==================================================
t_data = np.array([0, 1, 1, 0])

XOR_obj = LogicGateDL("XOR", x_data, t_data)
XOR_obj.train()

# prediction
print(XOR_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(data)
    print(sigmoid_val, " = ", logical_val)