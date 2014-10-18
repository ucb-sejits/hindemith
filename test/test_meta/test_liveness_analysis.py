from hindemith.meta.liveness_analysis import perform_liveness_analysis
from ctree.frontend import get_ast
from hindemith.meta.basic_blocks import get_basic_block, separate_composable_blocks
import unittest
import numpy as np

from .array_add import array_add


class TestLivenessAnalysis(unittest.TestCase):
    def test_simple(self):
        def func(a, b):
            return a + b

        a = 2
        b = 3
        ast = get_ast(func)
        basic_block = get_basic_block(ast)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(locals(), **globals()))
        basic_block = perform_liveness_analysis(basic_block)
        print(basic_block)
        self.assertEqual(basic_block[0].live_ins, {"a", "b"})
        self.assertEqual(basic_block[0].live_outs, set())

    def test_multi_block(self):
        def func(a, b):
            c = array_add(a, b)
            return c

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        ast = get_ast(func)
        basic_block = get_basic_block(ast)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(locals(), **globals()))
        basic_block = perform_liveness_analysis(basic_block)
        self.assertEqual(basic_block[0].live_ins, {"a", "b"})
        self.assertEqual(basic_block[0].live_outs, {"c"})
        self.assertEqual(basic_block[1].live_ins, {"c"})
        self.assertEqual(basic_block[1].live_outs, set())

    def test_complex_multi_block(self):
        def func(a, b):
            c = array_add(a, b)
            d = array_add(c, b)
            return 3 * d

        a = np.random.rand(256, 256).astype(np.float32) * 100
        b = np.random.rand(256, 256).astype(np.float32) * 100
        ast = get_ast(func)
        basic_block = get_basic_block(ast)
        basic_block = separate_composable_blocks(basic_block,
                                                 dict(locals(), **globals()))
        basic_block = perform_liveness_analysis(basic_block)
        self.assertEqual(basic_block[0].live_ins, {"a", "b"})
        self.assertEqual(basic_block[0].live_outs, {"d"})
        self.assertEqual(basic_block[1].live_ins, {"d"})
        self.assertEqual(basic_block[1].live_outs, set())
