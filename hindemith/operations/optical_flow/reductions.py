from ctree.jit import ConcreteSpecializedFunction, LazySpecializedFunction
from ctree.transformations import PyBasicConversions
from hindemith.operations.dense_linear_algebra import Array
import numpy as np
import ast
from ctree.frontend import get_ast
import astdump
import inspect
import textwrap
from ctypes import *
from ctree.c.nodes import *
from hindemith.utils import unique_kernel_name
from ctree.ocl.macros import *
from ctree.visitors import NodeTransformer

class ReduceConcrete(ConcreteSpecializedFunction):
    pass

class ReduceLazy(LazySpecializedFunction):
    def args_to_subconfig(self,args):
        size = args[0].size
        type = args[0].dtype
        return (size,type)
    def transform(self,tree, program_config):
        size = program_config[0][0]
        reducefunc = program_config[0][1]
        converter = PyBasicConversions()
        type_dict = {
            np.float32: c_float(),
            np.int32: c_int()
        }
        type = type_dict[type]
        reducefunc = converter.visit(reducefunc)
        reducefunc.return_type = type
        reducefunc.params[0].type = type
        reducefunc.params[1].type = type
        tree = [reducefunc]
        params = [
            SymbolRef('input',POINTER(type._ctype_)(),_global=True,_const=True),
            SymbolRef('temp',POINTER(type._ctype_)(),_global=False,_local=True)
        ]
        defn = []
        defn.extend([
            Assign(SymbolRef('gid',type),get_global_id(0)),
            Assign(SymbolRef('lid',type),get_local_id(0)),
            ast.parse("temp[lid] = input[lid]").body[0],
            For(Assign(SymbolRef()))
        ])

class Reduce(object):
    def __call__(self, array):
        return ReduceLazy(get_ast(self.func), array)
    def pure_python(self,array):
        return reduce(self.func, array.data.asarray)
    def func(self, x, y):
        raise NotImplementedError()

class Sum(Reduce):
    def func(self, x, y):
        return x+y

class Max(Reduce):
    def func(self, x, y):
        return x if x > y else y

class Min(Reduce):
    def func(x,y):
        return x if x < y else y