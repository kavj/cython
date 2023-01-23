import itertools
import operator

import Cython.Compiler.Nodes as nodes
import Cython.Compiler.ExprNodes as exprs
from Cython.Compiler.Visitor import TreeVisitor
from Cython.Compiler.ModuleNode import ModuleNode
import networkx as nx

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from typing import List, Optional, Tuple, Union


class expr_matcher:
    """
    Cython uses nodes that are not easily rendered hashable. The work-around is to do an exhaustive comparison
    and cache relationships where ids are known to be a match.

    This is meant to be used with atomic expr nodes, which should not be mutated.

    Note: needed for value tracking..
    """

    def __init__(self):
        self.known_matches = set()
        self.non_matches = set()
        self.negated_matches = set()
        self.negated_non_matches = set()

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


@dataclass(frozen=True)
class BasicBlock:
    statements: List[nodes.StatNode]
    label: int  # useful in case going by statement is too verbose
    depth: int
    predicate: Optional[exprs.AtomicExprNode]

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
    def is_terminated(self):
        # TODO: does cython allow return in parallel blocks?
        return isinstance(self.last, (nodes.BreakStatNode, nodes.ContinueStatNode, nodes.ReturnStatNode))

    @property
    def unterminated(self):
        return not self.is_terminated

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


class FlowGraph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.entry_block = None

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

    def remove_edge(self, source: BasicBlock, sink: BasicBlock):
        self.graph.remove_edge(source, sink)


class CFGBuilder:

    def __init__(self, start_from=0):
        self.loop_entry_points = []
        self.continue_map = defaultdict(list)
        self.break_map = defaultdict(list)
        self.scope_entry_blocks = {}  # map of header to initial entry blocks
        self.scope_exit_blocks = {}  # map of header to exit
        self.return_blocks = []
        self.labeler = itertools.count(start_from)
        self.graph = FlowGraph()
        self.counter = itertools.count()

    @property
    def entry_block(self):
        return self.graph.entry_block

    @property
    def next_label(self):
        return next(self.counter)

    @contextmanager
    def enclosing_loop(self, node: BasicBlock):
        assert node.is_loop_block
        self.loop_entry_points.append(node)
        yield
        self.loop_entry_points.pop()

    @property
    def depth(self):
        return len(self.loop_entry_points)

    def create_block(self, stmts: List[nodes.StatNode], start: int, stop: int, depth: int, predicate: Optional[exprs.AtomicExprNode]):
        label = next(self.labeler)
        assert 0 <= start <= stop
        if start == 0:
            if stop == len(stmts):
                # just wrap the statement list
                block = BasicBlock(stmts.copy(), label, depth, predicate)
            else:
                # split prefix
                interval = stmts[:stop]
                block = BasicBlock(interval, label, depth, predicate)
        else:
            if start < stop:
                pos = stmts[start].pos
                interval = stmts[start:stop]
            else:
                if start < len(stmts):
                    # not sure how this would happen..
                    pos = stmts[start].pos
                else:
                    pos = stmts[-1].pos
                interval = []
            block = BasicBlock(interval, label, depth, predicate)
        self.graph.graph.add_node(block)
        return block

    def insert_entry_block(self, entry_stmt: nodes.ParallelRangeNode):
        assert self.entry_block is None
        label = next(self.labeler)
        # this ensures blocks are consistent, even though we need to invent a fake statement list here
        entry = [entry_stmt]
        block = BasicBlock(entry, label, 0, None)
        self.graph.graph.add_node(block)
        self.graph.entry_block = block

    def add_edge(self, source: BasicBlock, sink: BasicBlock):
        if not isinstance(source, BasicBlock) or not isinstance(sink, BasicBlock):
            msg = f'Expected BasicBlock type for edge endpoints. Received "{source}" and "{sink}"'
            raise ValueError(msg)
        assert source is not sink
        self.graph.graph.add_edge(source, sink)

    def register_scope_entry_point(self, source: BasicBlock, sink: BasicBlock):
        assert source.is_entry_point
        self.add_edge(source, sink)
        self.scope_entry_blocks[source].append(sink)

    def register_scope_exit_point(self, source: BasicBlock, sink: BasicBlock):
        assert source.is_branch_entry_block or source.is_loop_block
        assert source not in self.scope_exit_blocks
        self.scope_exit_blocks[source] = sink

    def register_continue(self, block: BasicBlock):
        if not self.loop_entry_points:
            msg = f'Break: "{block.last}" encountered outside of loop.'
            raise ValueError(msg)
        self.continue_map[self.loop_entry_points[-1]].append(block)

    def register_break(self, block: BasicBlock):
        if not self.loop_entry_points:
            msg = f'Break: "{block.last}" encountered outside of loop.'
            raise ValueError(msg)
        self.break_map[self.loop_entry_points[-1]].append(block)

    def register_block_terminator(self, block: BasicBlock):
        last = block.last
        if last is not None:
            if isinstance(last, nodes.ContinueStatNode):
                self.register_continue(block)
            elif isinstance(last, nodes.BreakStatNode):
                self.register_break(block)
            elif isinstance(last, nodes.ReturnStatNode):
                self.return_blocks.append(block)


def matches_while_true(node: nodes.StatNode):
    if isinstance(node, nodes.WhileStatNode):
        if isinstance(node.test, exprs.BoolNode):
            if node.test.value is True:
                return True
    return False


def matches_negated(a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
    pass

def and_predicates(a: Union[exprs.AtomicExprNode, Tuple[exprs.AtomicExprNode, ...]], b: exprs.AtomicExprNode):

    if isinstance(a, tuple):
        pass
    else:
        pass


def or_predicates(a: exprs.AtomicExprNode, b: exprs.AtomicExprNode, pos):
    """
    This should correctly set up an "OR" node with folding if possible.
    It's needed to determine explicit predication.
    """
    return ()


def build_graph_recursive(statements: List[nodes.StatNode], builder: CFGBuilder, entry_point: BasicBlock, predicate: Optional[exprs.AtomicExprNode]):
    """
    This was adapted somewhat..

    It now receives a predicate, corresponding to the path entry that brought us here. This can be a more complicated expression than the one that appears in source,
    eg.
    if a > b:
        ...
    else:
        ...

    would have corresponding predicates

    1: a > b
    0: not (a > b)
    """
    prior_block = entry_point
    deferrals = []  # last_block determines if we have deferrals to this one
    for start, stop in sequence_block_intervals(statements):
        # TODO: we actually need to
        block = builder.create_block(statements, start, stop, builder.depth, None)
        if prior_block is entry_point:
            builder.add_edge(entry_point, block)
        elif prior_block.is_branch_entry_block:
            # If we have blocks exiting a branch, which do not contain a terminating statement
            # then add incoming edges to this block
            for d in deferrals.pop():
                builder.add_edge(d, block)
        else:
            if prior_block.is_loop_block:
                # indicate loop exit block so that breaks can be connected
                builder.register_scope_exit_point(prior_block, block)
            # loop or normal must add edge
            if prior_block.unterminated and not matches_while_true(prior_block.last):
                builder.add_edge(prior_block, block)
        # buffering taken care of by sequence block
        if block.is_loop_block:
            # need to preserve entry point info here..
            loop_header_stmt = statements[start]
            with builder.enclosing_loop(block):
                body = loop_header_stmt.body.stats
                first = block.first
                if isinstance(first, nodes.WhileStatNode):
                    if predicate is None:
                        predicate = first.condition
                    else:
                        predicate = make_and()
                last_interior_block = build_graph_recursive(body, builder, block, predicate)
                if last_interior_block.unterminated:
                    builder.add_edge(last_interior_block, block)
        elif block.is_branch_entry_block:
            raise NotImplementedError('Still converting branch types..')
            branch_exit_points = []
            if_stmt = statements[start]
            if_body = if_stmt.if_branch
            else_body = if_stmt.else_branch
            if_exit_block = build_graph_recursive(if_body, builder, block, predicate)
            # patch initial entry point
            if_entry_block, = builder.graph.successors(block)
            if if_exit_block.unterminated:
                branch_exit_points.append(if_exit_block)
            else_exit_block = build_graph_recursive(else_body, builder, block, predicate)
            for s in builder.graph.successors(block):
                assert isinstance(s, BasicBlock)
                if s is not if_entry_block:
                    break
            else:
                msg = f'No else entry block found for {statements[start]}'
                raise ValueError(msg)
            if else_exit_block.unterminated:
                branch_exit_points.append(else_exit_block)
            deferrals.append(branch_exit_points)
        elif block.is_terminated:
            builder.register_block_terminator(block)
        prior_block = block
    return prior_block


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


def build_graph(entry_point: nodes.ParallelRangeNode) -> FlowGraph:
    """
    This will construct a graph for a function or for loop. It will fail if continue or break appears here with respect
    to a loop that is not included in the graph.
    :param entry_point:
    :return:
    """
    builder = CFGBuilder()
    builder.insert_entry_block(entry_point)

    build_graph_recursive(entry_point.body.stats, builder, builder.entry_block, None)

    # Now clean up the graph
    for loop_header, continue_blocks in builder.continue_map.items():
        for block in continue_blocks:
            builder.add_edge(block, loop_header)

    for loop_header, break_blocks in builder.break_map.items():
        loop_exit_block = builder.scope_exit_blocks[loop_header]
        for block in break_blocks:
            builder.add_edge(block, loop_exit_block)

    graph = builder.graph
    remove_trivial_empty_blocks(graph)

    return graph



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
        super(OuterParallelGather, self).__init__()
        self.parallel_nodes = []

    def visit_Node(self, node):
        self.visitchildren(node)

    def visit_ParallelRangeNode(self, node):
        # if we encounter one of these, no need to visit children
        self.parallel_nodes.append(node)


class SubgraphCreation(TreeVisitor):
    def __init__(self):
        super(SubgraphCreation, self).__init__()

    def __call__(self, entry):
        if isinstance(entry, nodes.ParallelRangeNode):
            raise TypeError('did not expect this..')
        elif isinstance(entry, (nodes.CFuncDefNode, nodes.FuncDefNode, ModuleNode)):
            parallel_gather = OuterParallelGather()
            parallel_gather.visit(entry)
            parallel_entry_points = parallel_gather.parallel_nodes.copy()
            # make toy graph to start..
            if parallel_entry_points:
                graph = build_graph(parallel_entry_points[0])
                render_dot_graph(graph.graph, 'test_graph_conversion.dot', Path.cwd())
        return entry
