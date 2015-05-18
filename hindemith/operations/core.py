import numpy as np
from string import Template
from hindemith.types import hmarray
from ctree.frontend import get_ast
from ctree.transformations import PyBasicConversions
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
import ast


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
    def get_launcher(cls, sources, sinks, keywords, symbol_table):
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
    def emit(cls, sources, sinks, keywords, symbol_table):
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


class MapTransformer(ast.NodeTransformer):
    def __init__(self, mapping, target):
        super(MapTransformer, self).__init__()
        self.mapping = mapping
        self.target = target

    def visit_SymbolRef(self, node):
        if node.name in self.mapping:
            return StringTemplate(self.mapping[node.name].get_element())
        return node

    def visit_Return(self, node):
        value = self.visit(node.value)
        return C.Assign(StringTemplate(self.target.get_element()), value)


class Map(ElementLevel):
    """
    sink = fn(source1, source2, ...)
    """
    @classmethod
    def get_launch_parameters(cls, sources, sinks):
        num_work_items = np.prod(sources[0].shape)
        return (num_work_items, )

    @classmethod
    def emit(cls, sources, sinks, keywords, symbol_table):
        tree = get_ast(cls.fn)
        tree = PyBasicConversions().visit(tree)
        body = tree.body[0].defn
        mapping = {arg.name: source
                   for arg, source in zip(tree.body[0].params, sources)}
        visitor = MapTransformer(mapping, sinks[0])
        body = [visitor.visit(s) for s in body]
        return "\n".join([str(s) + ";" for s in body])
