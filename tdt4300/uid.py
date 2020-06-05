from itertools import count


def str_id(obj):
    return str(id(obj))


counter = count()


def inc_id():
    return next(counter)
