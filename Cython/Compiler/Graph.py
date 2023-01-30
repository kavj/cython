import itertools
import operator
import typing

import Cython.Compiler.Nodes as nodes
import Cython.Compiler.ExprNodes as exprs
from Cython.Compiler.Visitor import TreeVisitor
from Cython.Compiler.ModuleNode import ModuleNode
import networkx as nx

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import singledispatchmethod
from pathlib import Path

from typing import List, Optional, Tuple, Union


class expr_matcher:
    """
    Cython uses nodes that are not easily rendered hashable. This is bolted on to allow hashing, specifically
    when we

    This is meant to be used with atomic expr nodes, which should not be mutated.

    Note: needed for value tracking..
    """

    def __init__(self):
        self.known_matches = set()
        self.non_matches = set()
        self.negated_matches = set()
        self.negated_non_matches = set()



    def compute_expr_hash(self, expr: typing.Type[exprs.AtomicExprNode]):
        if not expr.subexprs:
            return

    def check_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        return frozenset((id(a), id(b))) in self.known_matches

    def check_non_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        return frozenset((id(a), id(b))) in self.non_matches

    def compute_negated_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        if isinstance(a, exprs.ConstNode) and isinstance(b, exprs.ConstNode):
            # catch and optimize this case
            pass
        if isinstance(a, exprs.NotNode):
            if self.compute_match(a.operand, b):
                self.insert_negated_match(a, b)
                return True
        if isinstance(b, exprs.NotNode):
            if self.compute_match(b.operand, a):
                self.insert_negated_match(a, b)
                return True
        self.insert_negated_non_match(a, b)
        return False

    def insert_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        self.known_matches.add(frozenset((id(a), id(b))))

    def insert_non_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        self.non_matches.add(frozenset((id(a), id(b))))

    def insert_negated_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        self.negated_matches.add(frozenset((id(a), id(b))))

    def insert_negated_non_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        self.negated_non_matches.add(frozenset((id(a), id(b))))

    def compute_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        if isinstance(a, exprs.NameNode):
            if isinstance(b, exprs.NameNode):
                if a.name == b.name:
                    self.insert_match(a, b)
                    return True
            self.insert_non_match(a, b)
            return False
        elif isinstance(a, exprs.ConstNode):
            if isinstance(b, exprs.ConstNode):
                if a.value == b.value:
                    self.insert_match(a, b)
                    return True
            self.insert_non_match(a, b)
            return False
        # If b matches one of the passed types, then it can't be a match
        elif isinstance(b, (exprs.NameNode, exprs.ConstNode)):
            self.insert_non_match(a, b)
            return False
        if type(a) != type(b):
            self.insert_non_match(a, b)
            return False
        subexpr_nodes_a = a.subexpr_nodes()
        subexpr_nodes_b = b.subexpr_nodes()
        if len(subexpr_nodes_a) != len(subexpr_nodes_b):
            # Not sure if this can happen without bugs.. maybe log a warning?
            self.insert_non_match(a, b)
            return False
        return all(self.compute_match(subexpr_a, subexpr_b) for (subexpr_a, subexpr_b) in zip(subexpr_nodes_a, subexpr_nodes_b))


def sequence_block_intervals(stmts: List[nodes.StatNode]):
    """
    For reversal, cast to list and reverse.
    This will intentionally yield a blank interval at the end and between any 2 consecutive scope points.
    :param stmts:
    :return:
    """

    block_start = 0
    block_last = -1
    for block_last, stmt in enumerate(stmts):
        if isinstance(stmt, (nodes.IfStatNode, nodes.ForInStatNode, nodes.WhileStatNode, nodes.ParallelRangeNode)):
            if block_start < block_last:
                yield block_start, block_last
            next_start = block_last + 1
            yield block_last, next_start
            block_start = next_start
    # always add an end block. Either it's non-empty or we need a reconvergence block
    yield block_start, block_last + 1


@dataclass
class BasicBlock:
    # Block
    label: int  # useful in case going by statement is too verbose
    depth: int
    # predicate consists of an optional expression and a boolean qualifier, which indicates whether it is flipped
    # Any not qualifiers should be stripped and placed in the second operand. This helps compare predicate
    # sequences for redundant branches or those that claim all remaining conditions, when this is actually decidable.
    predicate: Union[exprs.AtomicExprNode, bool] = True
    statements: List[nodes.StatNode] = field(default_factory=list)

    def append(self, stmt: nodes.StatNode):
        self.statements.append(stmt)

    @property
    def first(self) -> Optional[nodes.StatNode]:
        if self.statements:
            return self.statements[0]

    @property
    def last(self) -> Optional[nodes.StatNode]:
        if self.statements:
            return self.statements[-1]

    @property
    def is_loop_block(self):
        return isinstance(self.first, (nodes.ParallelRangeNode, nodes.ForInStatNode, nodes.WhileStatNode))

    @property
    def is_branch_entry_block(self):
        return isinstance(self.first, nodes.IfStatNode)

    @property
    def terminated(self):
        # TODO: does cython allow return in parallel blocks?
        return isinstance(self.last, (nodes.BreakStatNode, nodes.ContinueStatNode, nodes.ReturnStatNode))

    @property
    def unterminated(self):
        return not self.terminated

    @property
    def is_entry_point(self):
        # TODO: add support for others..
        return isinstance(self.first, (nodes.ForInStatNode, nodes.WhileStatNode, nodes.IfStatNode, nodes.ParallelRangeNode))

    @property
    def list_id(self):
        return id(self.statements)

    def __bool__(self):
        return operator.truth(self.statements)

    def __len__(self):
        return len(self.statements)

    def __iter__(self):
        return iter(self.statements)

    def __reversed__(self):
        return reversed(self.statements)

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        if self:
            return f'block {self.label} {self.depth} {str(self.first)}'
        else:
            return f'block {self.label} {self.depth}'


@dataclass
class loop_context:
    header: BasicBlock
    exit_block: BasicBlock


class FlowGraph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entry_block = None

    def add_node(self, block: BasicBlock, parents: List[BasicBlock]):
        self.graph.add_node(block)
        for p in parents:
            self.graph.add_edge(p, block)

    def add_entry_block(self, entry_block: BasicBlock):
        assert self.entry_block is None
        self.entry_block = entry_block

    @property
    def func_name(self):
        return self.entry_block.first.name

    def nodes(self):
        return self.graph.nodes()

    def reachable_nodes(self):
        return nx.dfs_preorder_nodes(self.graph, self.entry_block)

    def walk_nodes(self):
        for block in self.graph.nodes():
            for stmt in block:
                yield stmt

    def predecessors(self, node: BasicBlock):
        return self.graph.predecessors(node)

    def successors(self, node: BasicBlock):
        return self.graph.successors(node)

    def in_degree(self, block: Optional[BasicBlock] = None):
        if block is None:
            return self.graph.in_degree()
        return self.graph.in_degree(block)

    def out_degree(self, block: Optional[BasicBlock] = None):
        if block is None:
            return self.graph.out_degree()
        return self.graph.out_degree(block)

    def add_edge(self, source: BasicBlock, sink: BasicBlock):
        self.graph.add_edge(source, sink)

    def remove_edge(self, source: BasicBlock, sink: BasicBlock):
        self.graph.remove_edge(source, sink)


class CFGBuilder(TreeVisitor):
    def __init__(self, start_from=0):
        super().__init__()
        self.entry_points = []
        self.labeler = itertools.count(start_from)
        self.graph = FlowGraph()
        self.counter = itertools.count()
        self.current_block = None
        self.loop_depth = 0

    @property
    def entry_block(self):
        return self.graph.entry_block

    @property
    def next_label(self):
        return next(self.counter)

    @contextmanager
    def branch_scope(self, block: BasicBlock):
        self.entry_points.append(block)
        yield
        self.entry_points.pop()

    def handle_loop(self, node: Union[nodes.ParallelRangeNode, nodes.LoopNode]):
        header_block = self.add_block(node)
        exit_block = BasicBlock(self.next_label, self.loop_depth)
        self.graph.add_edge(header_block, exit_block)
        context = loop_context(header_block, exit_block)
        self.entry_points.append(context)
        self.loop_depth += 1
        self.visit(node.body)
        if not self.current_block.terminated:
            # make back edge
            self.graph.add_edge(self.current_block, header_block)

    def add_block(self, node: Optional[nodes.StatNode] = None, predicate: Optional[exprs.AtomicExprNode] = None):
        block = BasicBlock(self.next_label, self.loop_depth, predicate=predicate)
        if self.entry_block is None:
            self.graph.add_entry_block(block)
        else:
            parents = [] if self.current_block is None else [self.current_block]
            self.graph.add_node(block, parents)
        if node is not None:
            block.append(node)
        self.current_block = block
        return block

    def visit_ParallelRangeNode(self, node: nodes.ParallelRangeNode):
        self.handle_loop(node)

    def visit_IfStatNode(self, node: nodes.IfStatNode):
        header_block = self.add_block(node)
        deferrals = []
        for branch in node.if_clauses:
            self.current_block = header_block
            self.visit(branch)
            if self.current_block.unterminated:
                deferrals.append(self.current_block)
        if node.else_clause is not None:
            self.current_block = header_block
            self.visit(node.else_clause)
            if self.current_block.unterminated:
                deferrals.append(self.current_block)
        exit_block = self.add_block()
        for d in deferrals:
            self.graph.add_edge(d, exit_block)

    def visit_IfClauseNode(self, node: nodes.IfClauseNode):
        self.add_block(predicate=node.condition)
        self.visit(node.body)

    def visit_ForInStatNode(self, node: nodes.ForInStatNode):
        self.handle_loop(node)

    def visit_WhileStatNode(self, node: nodes.WhileStatNode):
        self.handle_loop(node)

    def visit_BreakStatNode(self, node: nodes.BreakStatNode):
        self.current_block.append(node)
        self.graph.add_edge(self.current_block, self.entry_points[-1].exit_block)

    def visit_StatNode(self, node: nodes.StatNode):
        if not self.current_block.terminated:
            self.current_block.append(node)

    def visit_StatListNode(self, node: nodes.StatListNode):
        for stmt in node.stats:
            self.visit(stmt)


def matches_while_true(node: nodes.StatNode):
    if isinstance(node, nodes.WhileStatNode):
        if isinstance(node.test, exprs.BoolNode):
            if node.test.value is True:
                return True
    return False


def matches_negated(a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
    pass

def and_predicates(a: Tuple[Union[exprs.AtomicExprNode, bool], bool], b: Tuple[Union[exprs.AtomicExprNode, bool], bool]):
    return ()


def or_predicates(a: exprs.AtomicExprNode, b: exprs.AtomicExprNode, pos):
    """
    This should correctly set up an "OR" node with folding if possible.
    It's needed to determine explicit predication.
    """
    return ()


def iter_statements(node: Union[nodes.StatNode, nodes.StatListNode], reverse=False):
    if isinstance(node, nodes.StatListNode):
        stats = node.stats
    else:
        stats = node.body.stats
    return reversed(stats) if reverse else iter(stats)


def remove_trivial_empty_blocks(graph: FlowGraph):
    """
    merge blocks with single in and out degree into their predecessor if the predecessor
    has out degree one and is not a control flow entry point.
    :param graph:
    :return:
    """

    # Since this is only a view of the underlying tree, only merge empty blocks.
    worklist = [node for node in graph.nodes() if not node
                and graph.in_degree(node) == 1 and graph.out_degree(node) == 1]

    while worklist:
        node = worklist.pop()
        predecessor, = graph.predecessors(node)
        if graph.out_degree(predecessor) == 1 and not predecessor.is_entry_point:
            successor, = graph.successors(node)
            graph.graph.remove_node(node)
            graph.graph.add_edge(predecessor, successor)



def get_loop_exit_block(graph: FlowGraph, node: BasicBlock) -> Optional[BasicBlock]:
    """
    This will return a loop exit if there's an edge from the header block to some block outside the loop.
    Otherwise returns None.

    :param graph:
    :param node:
    :return:
    """
    assert node.is_loop_block
    for block in graph.successors(node):
        if block.depth == node.depth:
            return block


def dominator_tree(graph: FlowGraph):
    idoms = nx.dominance.immediate_dominators(graph.graph, graph.entry_block)
    # remove self dominance
    g = nx.DiGraph()
    for k, v in idoms.items():
        if k.label != v.label:
            g.add_edge(v, k)
    return g


def render_dot_graph(graph: nx.DiGraph, name: str, out_path: Path):
    dot_graph = nx.drawing.nx_pydot.to_pydot(graph)
    img_name = f'{name}.png'
    render_path = out_path.joinpath(img_name)
    dot_graph.write_png(render_path)


def render_dominator_tree(graph: FlowGraph, out_path: Path):
    """
    convenience method to render a dominator tree
    :param graph:
    :param out_path:
    :return:
    """
    dom_tree = dominator_tree(graph)
    img_name = f'{graph.func_name}_doms'
    render_dot_graph(dom_tree, img_name, out_path)


class OuterParallelGather(TreeVisitor):
    """
    Find out if a node appears in a subtree.
    """

    def __init__(self):
        super().__init__()
        self.parallel_nodes = []

    def visit_Node(self, node):
        self.visitchildren(node)

    def visit_ParallelRangeNode(self, node):
        # if we encounter one of these, no need to visit children
        self.parallel_nodes.append(node)


class SubgraphCreation(TreeVisitor):
    def __init__(self):
        super().__init__()

    def __call__(self, entry):
        if isinstance(entry, nodes.ParallelRangeNode):
            raise TypeError('did not expect this..')
        elif isinstance(entry, (nodes.CFuncDefNode, nodes.FuncDefNode, ModuleNode)):
            parallel_gather = OuterParallelGather()
            parallel_gather.visit(entry)
            parallel_entry_points = parallel_gather.parallel_nodes.copy()
            # make toy graph to start..
            if len(parallel_entry_points) == 1:
                builder = CFGBuilder()
                builder.visit(parallel_entry_points[0])
                graph = builder.graph
                render_dot_graph(graph.graph, 'test_graph_conversion.dot', Path.cwd())
        return entry
