__author__ = 'leonardtruong'

nameCnt = 0


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
