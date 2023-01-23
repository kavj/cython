from collections import  deque
from typing import Generator, Union

import Cython.Compiler.Nodes as nodes
import Cython.Compiler.ExprNodes as exprs


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
