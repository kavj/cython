import itertools

from collections import defaultdict, deque
from typing import Generator, Iterator, List, Union

import Cython.Compiler.Nodes as nodes
import Cython.Compiler.ExprNodes as exprs


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

    def check_match(self, a: exprs.AtomicExprNode, b: exprs.AtomicExprNode):
        id_a = id(a)
        id_b = id(b)
        if (id_a, id_b) in self.known_matches:
            return True
        elif (id_a, id_b) in self.non_matches:
            return False
        sig = (id_a, id_b)
        reverse_sig = (id_b, id_a)
        if isinstance(a, exprs.NameNode):
            if isinstance(b, exprs.NameNode):
                if a.name == b.name:
                    self.known_matches.add(sig)
                    self.known_matches.add(reverse_sig)
                    return True
            self.non_matches.add(sig)
            self.non_matches.add(reverse_sig)
            return False
        elif isinstance(a, exprs.ConstNode):
            if isinstance(b, exprs.ConstNode):
                if a.value == b.value:
                    self.known_matches.add(sig)
                    self.known_matches.add(reverse_sig)
                    return True
            self.non_matches.add(sig)
            self.non_matches.add(reverse_sig)
            return False
        # If b matches one of the passed types, then it can't be a match
        elif isinstance(b, (exprs.NameNode, exprs.ConstNode)):
            self.non_matches.add(sig)
            self.non_matches.add(reverse_sig)
            return False
        if type(a) != type(b):
            self.non_matches.add(sig)
            self.non_matches.add(reverse_sig)
            return False
        subexpr_nodes_a = a.subexpr_nodes()
        subexpr_nodes_b = b.subexpr_nodes()
        if len(subexpr_nodes_a) != len(subexpr_nodes_b):
            # Not sure if this can happen without bugs.. maybe log a warning?
            self.non_matches.add(sig)
            self.non_matches.add(reverse_sig)
            return False
        return all(self.check_match(subexpr_a, subexpr_b) for (subexpr_a, subexpr_b) in zip(subexpr_nodes_a, subexpr_nodes_b))


def walk(node: exprs.AtomicExprNode):
    """
    yields all distinct sub-expressions and the base expression
    It was changed to include the base expression so that it's safe
    to walk without the need for explicit dispatch mechanics in higher level
    functions on Expression to disambiguate Expression vs non-Expression value
    references.

    :param node:
    :return:
    """
    if not isinstance(node, exprs.AtomicExprNode):
        msg = f'walk expects a value ref. Received: "{node}"'
        raise TypeError(msg)
    if node.subexprs:
        enqueued = [(node, node.subexpr_nodes())]
        while enqueued:
            expr, subexprs = enqueued[-1]
            while subexprs:
                subexpr = subexprs.pop(0)
                if subexpr.subexprs:
                    enqueued.append((subexpr, subexprs.subexpr_nodes()))
                    break
                yield subexpr
            else:
                yield expr
                enqueued.pop()
    else:
        # class doesn't declare sub-expressions
        yield node


def walk_parameters(node: exprs.AtomicExprNode):
    for value in walk(node):
        if isinstance(value, exprs.NameNode):
            yield value


def get_statement_lists(node: Union[nodes.StatListNode, nodes.LoopNode, nodes.IfStatNode],
                        enter_loops=True) -> Generator[nodes.StatListNode, None, None]:
    """
    yields all statement lists by pre-ordering, breadth first
    :param node:
    :param enter_loops:
    :return:
    """

    queued = deque()
    if isinstance(node, nodes.IfStatNode):
        queued.append(node.else_branch)
        queued.append(node.if_branch)
    elif isinstance(node, (nodes.ForInStatNode, nodes.WhileStatNode)):
        queued.append(node.body)
    else:  # statement list
        queued.append(node)
    while queued:
        stmts = queued.pop()
        # yield in case caller modifies trim
        yield stmts
        for stmt in stmts:
            if isinstance(stmt, nodes.IfStatNode):
                queued.appendleft(stmt.if_branch)
                queued.appendleft(stmt.else_branch)
            elif enter_loops and isinstance(node, (nodes.ForInStatNode, nodes.WhileStatNode)):
                queued.appendleft(stmt.body)
