class Person:

    count = 0

    def __init__(self, name):
        self.name = name

    @classmethod
    def getCount(cls):
        return cls.count

