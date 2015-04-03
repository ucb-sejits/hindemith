operations = []


class HMLevel(object):
    def compile(self):
        raise NotImplementedError()


class DeviceLevel(HMLevel):
    pass


class BlockLevel(HMLevel):
    pass


class ItemLevel(HMLevel):
    pass


class ElementLevel(HMLevel):
    pass


def register_operation(op):
    operations.append(op)
