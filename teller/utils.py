__author__ = 'leonardtruong'

nameCnt = 0


def unique_name():
    global nameCnt
    name = '_f{0}'.format(nameCnt)
    nameCnt += 1
    return name


def clamp(val, minimum, maximum):
    return max(minimum, min(val, maximum))
