# from .basic_blocks import ComposableBlock, NonComposableBlock
import ast


class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.gen = set()
        self.kill = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.kill:
                self.gen.add(node.id)
        else:
            self.kill.add(node.id)


def perform_liveness_analysis(basic_block):
    for index, sub_block in enumerate(reversed(basic_block)):
        analyzer = Analyzer()
        for statement in sub_block:
            analyzer.visit(statement)
        if index == len(basic_block):
            sub_block.live_outs = set()
        else:
            sub_block.live_outs = set().union(
                *(block.live_ins for block
                  in basic_block[len(basic_block) - index:]))
        sub_block.live_ins = analyzer.gen.union(
            sub_block.live_outs.difference(analyzer.kill))
        sub_block.kill = analyzer.kill
    return basic_block
