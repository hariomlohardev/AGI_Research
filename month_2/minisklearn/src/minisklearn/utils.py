"""
All the utills functions are pressent in this file :)

"""


def dot_product(a: list, b: list):
    output = sum(a_i * b_i for a_i, b_i in zip(a, b))
    return output


def if_empty_raise(data: list, message=""):
    if len(data) == 0:
        raise ValueError(message)
