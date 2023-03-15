import numpy as np
from LogicGate import LogicGate

x_data = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])

# AND Gate ==================================================
t_data = np.array([0, 0, 0, 1])

AND_obj = LogicGate("AND_GATE", x_data, t_data)
AND_obj.train()

# prediction
print(AND_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = AND_obj.predict(data)
    print(data, " = ", logical_val)

# OR Gate ==================================================
t_data = np.array([0, 1, 1, 1])

OR_obj = LogicGate("OR_GATE", x_data, t_data)
OR_obj.train()

# prediction
print(OR_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = OR_obj.predict(data)
    print(data, " = ", logical_val)

# NAND Gate ==================================================
t_data = np.array([1, 1, 1, 0])

NAND_obj = LogicGate("NAND_GATE", x_data, t_data)
NAND_obj.train()

# prediction
print(NAND_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = NAND_obj.predict(data)
    print(data, " = ", logical_val)

# XOR Gate ==================================================
t_data = np.array([0, 1, 1, 0])

XOR_obj = LogicGate("XOR_GATE", x_data, t_data)

# XOR Gate를 보면, 손실함수 값이 2.7 근처에서 더 이상 감소하지 않는 것을 볼수 있음
XOR_obj.train()

# prediction
# XOR Gate prediction => 예측이 되지 않음
print(XOR_obj.name)
for data in x_data:
    (sigmoid_val, logical_val) = XOR_obj.predict(data)
    print(data, " = ", logical_val)

s1 = [] # NAND 출력
s2 = [] # OR 출력

new_input_data = [] # AND 입력
final_output = [] # AND 출력

for index in range(len(x_data)):
    s1 = NAND_obj.predict(x_data[index]) # NAND 출력
    s2 = OR_obj.predict(x_data[index]) # OR 출력

    new_input_data.append(s1[-1]) # AND 입력
    new_input_data.append(s2[-1]) # AND 입력

    (sigmoid_val, logical_val) = AND_obj.predict(np.array(new_input_data))

    final_output.append(logical_val) # AND 출력, 즉 XOR 출력
    new_input_data = [] # AND 입력 초기화

print(XOR_obj.name)
for index in range(len(x_data)):
    print(x_data[index], " = ", final_output[index])