a = [12, 20, 30, 40, 50]
b = (12, 20, 30, 40, 50)
c = {"kim": 90, "lee": 100, "fuck": 70}
d = "string"
e = [[12, 20, 30], [12, 20, 30], [12, 20, 30], 40, 50]
lc = list(c.items())

print(type(a))
print(lc)
print(len(d))

if "kim" in c:
    print("fuck")
else:
    print("fff")

for data in range(10):
    print(data, " ", end="")
print()

for data in a:
    print(data, " ", end="")
print()

for key, value in c.items():
    print(key, value, sep=" ")
print()

f = [x * 2 for x in range(5)]
print(f)


def sum(x, y):
    return x + y


print(sum(10, 20))


def multi_ret_func(x):
    return x + 1, x + 2, x + 3


r = multi_ret_func(10)
print(r[0], r[2])


# default parameter
def print_name(name, count=2):
    for i in range(count):
        print("name: ", name)


print_name("DAVE")


# mutable, immutable parameter
def mutable_immutable_func(int_x, input_list):
    int_x += 1
    input_list.append(100)


x = 1
test_list = [1, 2, 3, ]

mutable_immutable_func(x, test_list)
print("x: ", x, ", test_list: ", test_list)

# lambda function
g = lambda x: x + 100

for i in range(3):
    print(g(i))


# lambda 에서 입력 값을 반드시 이용할 필요는 없음
def print_hello():
    print("HELLO WORLD")


def test_lambda(s, t):
    print("input1: ", s, ", input2: ", t)


s = 100
t = 200

fx = lambda x, y: test_lambda(s, t)
fy = lambda x, y: print_hello()

fx(500, 1000)
fy(300, 600)
