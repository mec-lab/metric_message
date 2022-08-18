import sympy as sp
import numpy as np

def get_complexity(expression):
        
    tree_nodes = [element for element in sp.preorder_traversal(sp.simplify(expression))]
    return len(tree_nodes)

def get_shannon_diversity(expression):
    # H = -Σpi * ln(pi)    

    simple = sp.simplify(expression)
    nodes = [str(elem) for elem in sp.preorder_traversal(simple)]
    unique = list(set(nodes))

    frequencies = [(elem == np.array(nodes)).sum() for elem in nodes]
    p = frequencies / np.sum(frequencies)

    h = -np.sum(p * np.log(p))

    return h            
       
