from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.transformations import PyBasicConversions
import numpy as np
import ast
from ctree.frontend import get_ast
import inspect
import textwrap
from ctypes import *
from ctree.c.nodes import *
from ctree.ocl.macros import *
from hindemith.types.common import Array
try:
    from functools import reduce
except:
    pass

# class ReduceConcrete(ConcreteSpecializedFunction):
#     pass
#
class ReduceLazy(LazySpecializedFunction):
    pass
#     def args_to_subconfig(self,args):
#         size = args[0].size
#         type = args[0].dtype
#         return (size,type)
#     def transform(self,tree, program_config):
#         size = program_config[0][0]
#         reducefunc = program_config[0][1]
#         converter = PyBasicConversions()
#         type_dict = {
#             np.float32: c_float(),
#             np.int32: c_int()
#         }
#         type = type_dict[type]
#         reducefunc = converter.visit(reducefunc)
#         reducefunc.return_type = type
#         reducefunc.params[0].type = type
#         reducefunc.params[1].type = type
#         tree = [reducefunc]
#         params = [
#             SymbolRef('input',POINTER(type._ctype_)(),_global=True,_const=True),
#             SymbolRef('temp',POINTER(type._ctype_)(),_global=False,_local=True),
#             SymbolRef('output',POINTER(type._ctype_)(),_global=True)
#         ]
#         defn = []
#         defn.extend([
#             Assign(SymbolRef('gid',c_int()),get_global_id(0)),
#             Assign(SymbolRef('lsize',c_int()),get_local_size(0)),
#             Assign(SymbolRef('lid',c_int()),get_local_id(0)),
#             Assign(ArrayRef(SymbolRef('temp'),SymbolRef('lid')),ArrayRef(SymbolRef('input'),SymbolRef('gid'))),
#             barrier(CLK_LOCAL_MEM_FENCE()),
#             For(Assign(SymbolRef('i'),c_int()),Lt(SymbolRef('i'),Constant(0)),Assign(SymbolRef('i'),BitShL(SymbolRef('i'),Constant(0))),
#                 body=[
#                     If(Lt(SymbolRef('lid'),SymbolRef('i')),
#
#                     )
#                 ]
#             )
#         ])

class Reduce(object):
    def __init__(self,pure_python = False):
        self.pure_python = pure_python
    def __call__(self, array):
        if(self.pure_python):
            return reduce(self.func, array.data.flat)
        else:
            return reduce(self.func, array.data.flat)
    def func(x, y):
        raise NotImplementedError()

## Functions

class Sum(Reduce):
    @staticmethod
    def func(x, y):
        return x+y

class Max(Reduce):
    @staticmethod
    def func(x, y):
        return x if x > y else y

class Min(Reduce):
    @staticmethod
    def func(x,y):
        return x if x < y else y
