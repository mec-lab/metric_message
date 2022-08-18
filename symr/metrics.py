import sympy as sp
import numpy as np
import apted

def compute_complexity(expression):
        
    tree_nodes = [element for element in sp.preorder_traversal(sp.simplify(expression))]
    return len(tree_nodes)

def compute_shannon_diversity(expression):
    # H = -Σpi * ln(pi)    

    simple = sp.simplify(expression)
    nodes = [str(elem) for elem in sp.preorder_traversal(simple)]
    unique = list(set(nodes))

    frequencies = [(elem == np.array(nodes)).sum() for elem in nodes]
    p = frequencies / np.sum(frequencies)

    h = -np.sum(p * np.log(p))

    return h            
       

def compute_exact_equivalence(expression_a, expression_b):
    """
    symbolic metric

    reported in:
        Petersen _et al._ 2019
        La Cava _et al._ 2021
    """
    
    simple_expression_a = sp.simplify(expression_a)
    simple_expression_b = sp.simplify(expression_b)

    difference = sp.simplify(simple_expression_a - simple_expression_b)

    return 0

def compute_tree_distance(expression_a, expression_b):
    """
    symbolic metric
    
    reported in:
        
    """
   
    return 0 

def compute_r2(expression_a, expression_b, inputs):
    """
    numerical metric
    
    """
    pass

def compute_truncated_r2(expression_a, expression_b, inputs):
    """
    numerical metric
    
    reported in:
        Kamienny _et al._ 2022
    """
    pass

def compute_relative_error(expression_a, expression_b, inputs):
    """
    numerical metric

    relative absolute error

    reported in:
        Vastl _et al._ 2022
    """
    pass
    
def compute_isclose(expression_a, expression_b, inputs): 
    """
    numerical metric (ad-hoc accuracy proxy)

    reported in:
        Biggio _et al._ 2021
    """
    pass


def compute_r2_over_threshold(expression_a, expression_b, inputs, threshold):
    """
    numerical metric (ad-hoc accuracy proxy)

    reported in:
        Kamienny _et al._ 2022 (with threshold 0.99)
        Biggio _et al._ 2019 (with threshold 0.95)
    """
    pass
