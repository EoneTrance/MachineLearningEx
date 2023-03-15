class Person:

    count = 0

    def __init__(self, name):
        self.__name = name

    def __printName(self):
        print(self.name)
        print(Person.count)

person = Person("LEE")
