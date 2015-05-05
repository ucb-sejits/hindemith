import numpy as np
from string import Template
from hindemith.types import hmarray


class HMUndefinedMethodError(NotImplementedError):
    def __init__(self, cls, method):
        message = "{} operation must implement a {} method".format(cls, method)
        super(HMUndefinedMethodError, self).__init__(message)


class HMOperation(object):
    pass


class DeviceLevel(HMOperation):
    """
    An operation that can run multiple OpenCL kernels without interference.
    """
    @classmethod
    def get_launcher(cls, sources, sinks, symbol_table):
        raise HMUndefinedMethodError(cls, "get_launcher")


class BlockLevel(HMOperation):
    """
    An OpenCL Kernel
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        """
        Return a tuple of parameters used to launch the kernel

        :return: (num_work_items, )
        :rtype: tuple(int)
        """
        raise HMUndefinedMethodError(cls, "get_launch_parameters")

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        """
        Emit the code to be the body of the OpenCL Kernel.  A
        BlockLevel operation can use any valid OpenCL api calls and
        constructs.

        :param list sources: List of sources as strings
        :param list sinks: List of sinks as strings
        :param dict symbol_table: The current symbol_table

        :returns: String to be appended to kernel body
        :rtype: str
        """
        raise HMUndefinedMethodError(cls, "emit")


class ElementLevel(HMOperation):
    """
    An operation that does not communicate across work items.
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        """
        Return a tuple of parameters used to launch the kernel

        :return: (num_work_items, )
        :rtype: tuple(int)
        """
        raise HMUndefinedMethodError(cls, "get_launch_parameters")

    @classmethod
    def emit(cls, sources, sinks, symbol_table):
        """
        Emit the code to be inserted into the body of the Kernel,
        ElementLevel operations are not allowed to communicate across
        work items (i.e. using barriers or local memory).  If done,
        behavior is undefined.

        :param list sources: List of sources as strings
        :param list sinks: List of sinks as strings
        :param dict symbol_table: The current symbol_table

        :returns: String to be appended to kernel body
        :rtype: str
        """
        raise HMUndefinedMethodError(cls, "emit")
