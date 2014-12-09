from ctree import util

__author__ = 'leonardtruong'

nameCnt = 0

import sys
from ctree.util import Timer


def unique_name():
    global nameCnt
    name = '_f{0}'.format(nameCnt)
    nameCnt += 1
    return name


kernelNameCnt = 0


def unique_kernel_name():
    global kernelNameCnt
    name = '_kernel{0}'.format(kernelNameCnt)
    kernelNameCnt += 1
    return name


pythonNameCnt = 0


def unique_python_name():
    global pythonNameCnt
    name = '_python_func{0}'.format(pythonNameCnt)
    pythonNameCnt += 1
    return name


def clamp(val, minimum, maximum):
    return max(minimum, min(val, maximum))


class TellerException(Exception):
    pass


class UnsupportedBackendError(TellerException):
    pass


class UnsupportedTypeError(TellerException):
    pass


def get_best_time(function, trials=3, iterations=100):
    best_time = sys.float_info.max
    for trial in xrange(trials):
        # print("Starting trial {}".format(trial))
        with Timer() as t:
            for iteration in xrange(iterations):
                # print("iteration {}".format(iteration))
                function()
        elapsed = t.interval / float(iterations)
        if elapsed < best_time:
            best_time = elapsed
    return best_time * 1000.0


