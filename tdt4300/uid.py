from itertools import count


def uid(obj):
    return str(id(obj))


counter = count()


def inc_id():
    return next(counter)
