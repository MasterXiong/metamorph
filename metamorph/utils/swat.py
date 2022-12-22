import numpy as np

from metamorph.config import cfg


def getChildrens(parents):
    childrens = []
    for cur_node_idx in range(len(parents)):
        childrens.append([])
        for node_idx in range(cur_node_idx, len(parents)):
            if cur_node_idx == parents[node_idx]:
                childrens[cur_node_idx].append(node_idx)
    return childrens


def lcrs(graph):
    new_graph = [[] for _ in graph]
    for node, children in enumerate(graph):
        if len(children) > 0:
            temp = children[0]
            new_graph[node].insert(0, temp)
            for sibling in children[1:]:
                new_graph[temp].append(sibling)
                temp = sibling
    return new_graph


def getTraversal(parents, traversal_types=cfg.MODEL.TRANSFORMER.TRAVERSALS):
    """Reconstruct tree and return a lists of node position in multiple traversals"""
    
    def postorder(children):
        trav = []
        def visit(node):
            for i in children[node]:
                visit(i)
            trav.append(node)
        visit(0)
        return trav

    def inorder(children):
        # assert binary tree
        trav = []
        def visit(node):
            if children[node]:
                visit(children[node][0])
            trav.append(node)
            if len(children[node]) == 2:
                visit(children[node][1])
        visit(0)
        return trav

    children = getChildrens(parents)
    traversals = []
    for ttype in traversal_types:
        if ttype == 'pre':
            indices = list(range(len(children)))
        else:
            if ttype == 'inlcrs':
                traversal = inorder(lcrs(children))
            elif ttype == 'postlcrs':
                traversal = postorder(lcrs(children))
            # indices = traversal
            indices = []
            for i in list(range(len(children))):
                indices.append(traversal.index(i))
        indices.extend([0 for _ in range(cfg.MODEL.MAX_LIMBS - len(indices))])
        traversals.append(indices)
    traversals = np.stack(traversals, axis=1)
    return traversals