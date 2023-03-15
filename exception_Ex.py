def calc(list_data):

    sum = 0

    try:
        sum = list_data[0] + list_data[1] + list_data[2]

        if sum < 0:
            raise Exception("Sum is minus")

    except IndexError as err:
        print(str(err))
    except Exception as err:
        print(str(err))
    finally:
        print(sum)
