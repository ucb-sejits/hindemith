import ast
import inspect
import textwrap
from hindemith.operations import HMOperation
from hindemith.cl import context, queue
from string import Template
import pycl as cl


def get_ast(obj):
    """
    Return the Python ast for the given object, which may
    be anything that inspect.getsource accepts (incl.
    a module, class, method, function, traceback, frame,
    or code object).
    """
    indented_program_txt = inspect.getsource(obj)
    program_txt = textwrap.dedent(indented_program_txt)
    return ast.parse(program_txt)


class Block(list):
    pass


class Kernel(object):
    def __init__(self, launch_parameters):
        self.launch_parameters = launch_parameters
        self.body = ""
        self.sources = set()
        self.sinks = set()

    def append_body(self, string):
        self.body += string

    def compile(self):
        sources = set(src.id for src in self.sources)
        sinks = set(src.id for src in self.sinks)
        params = sources | sinks
        self.params = list(params)
        params_str = ", ".join("global float* {}".format(p) for p in self.params)
        kernel = Template(
            """
            __kernel void fn($params) {
              if (get_global_id(0) < $num_work_items) {
                $body
              }
            }
            """
        ).substitute(params=params_str, body=self.body,
                     num_work_items=self.launch_parameters[0])
        kernel = cl.clCreateProgramWithSource(context, kernel).build()['fn']
        kernel.argtypes = tuple(cl.cl_mem for _ in self.params)
        self.kernel = kernel

    def launch(self, symbol_table):
        args = [symbol_table[p].ocl_buf for p in self.params]
        global_size = self.launch_parameters[0]
        if global_size % 32:
            padded = (global_size + 31) & ~0x20
        else:
            padded = global_size
        self.kernel(*args).on(queue, (padded,))


class Compose(object):
    unique_id = -1

    def __init__(self, func, symbol_table):
        self.symbol_table = symbol_table
        tree = get_ast(func)
        func_def = tree.body[0]
        new_body = []
        for statement in func_def.body:
            if self.is_hindemith_operation(statement):
                if len(new_body) < 1 or not isinstance(new_body[-1], Block):
                    new_body.append(Block())
                new_body[-1].append(statement)
            else:
                new_body.append(statement)
        processed = []
        for block_or_statement in new_body:
            if isinstance(block_or_statement, Block):
                processed.append(self.gen_hm_func(block_or_statement))
            else:
                processed.append(block_or_statement)
        func_def.body = processed
        func_def.decorator_list = []
        ast.fix_missing_locations(tree)
        exec(compile(tree, filename="<nofile>", mode="exec"),
             self.symbol_table, self.symbol_table)
        self.func_name = func_def.name

    def __call__(self, *args, **kwargs):
        self.symbol_table[self.func_name](*args, **kwargs)

    def gen_hm_func(self, block):
        sources = []
        sinks = []
        for op in block:
            _sinks, _sources = self.get_sinks_and_sources(op)
            sinks.extend(_sinks)
            sources.extend(_sources)

        def fn(*args, **kwargs):
            for source, arg in zip(sources, args):
                self.symbol_table[source.id] = arg
            kernels = []
            for op in block:
                _sinks, _sources = self.get_sinks_and_sources(op)
                launch_params = self.get_launch_params(op, _sources, _sinks)
                if len(kernels) < 1 or \
                   kernels[-1].launch_paramaters != launch_params:
                    kernels.append(Kernel(launch_params))
                else:
                    raise NotImplementedError()
                kernels[-1].append_body(self.get_emit(op, _sources, _sinks))
                kernels[-1].sources |= set(_sources)
                kernels[-1].sinks |= set(_sinks)
            for kernel in kernels:
                kernel.compile()
                kernel.launch(self.symbol_table)
            return (self.symbol_table[sink.id] for sink in sinks)

        self.unique_id += 1
        name = "_f{}".format(self.unique_id)
        self.symbol_table[name] = fn
        func = ast.Call(
            ast.Name(name, ast.Load()),
            sources,
            [],
            None,
            None,
        )
        if len(sinks) > 1:
            targets = [ast.Tuple(sinks, ast.Store())]
        else:
            targets = [sinks[0]]
        return ast.Assign(targets, func)

    def get_sinks_and_sources(self, operation):
        if isinstance(operation.targets[0], ast.Name):
            sources = [operation.targets[0]]
        else:
            sources = [elt for elt in operation.targets[0].elts]
        if isinstance(operation.value, ast.Call):
            sinks = [arg for arg in operation.value.args]
        else:
            raise NotImplementedError()
        return sources, sinks

    def get_emit(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [src.id for src in sources]
        sinks = [sink.id for sink in sinks]
        return func.emit(sources, sinks)

    def get_launch_params(self, operation, sources, sinks):
        func = self.eval_in_symbol_table(operation.value.func)
        sources = [self.eval_in_symbol_table(src) for src in sources]
        sinks = [self.eval_in_symbol_table(sink) for sink in sinks]
        return func.get_launch_parameters(sources, sinks)

    def eval_in_symbol_table(self, func):
        if isinstance(func, ast.Name):
            return self.symbol_table[func.id]
        elif isinstance(func, ast.Attribute):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def is_hindemith_operation(self, statement):
        if not isinstance(statement, ast.Assign):
            return False
        value = statement.value
        if isinstance(value, ast.Call):
            func = self.eval_in_symbol_table(value.func)
            if issubclass(func, HMOperation):
                return True
        return False


def compose(fn):
    symbol_table = {}
    frame = inspect.stack()[1][0]
    while frame is not None:
        symbol_table.update(frame.f_locals)
        symbol_table.update(frame.f_globals)
        frame = frame.f_back

    def wrapped(*args, **kwargs):
        return Compose(fn, symbol_table)(*args, **kwargs)
    return wrapped
