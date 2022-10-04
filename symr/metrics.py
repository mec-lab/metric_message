import sympy as sp
import numpy as np

import apted
from apted import APTED, PerEditOperationConfig
from apted.helpers import Tree

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

    return difference == 0

def tree_to_brackets(sp_tree):
    """
    helper function for converting from the parentheses-based
    trees used by sympy to bracket-based trees used by apted
    """

    brackets_tree = sp_tree.replace("(","{")
    brackets_tree = brackets_tree.replace(")","}")
    brackets_tree = brackets_tree.replace(",","}{")
    brackets_tree = brackets_tree.replace(" ","")

    return "{" + brackets_tree + "}"

def compute_tree_distance(expression_a, expression_b):
    """
    symbolic metric
    
    compare two equations in terms of tree edit distance

    args:
        str_eqn1    equation 1, in a string that can be parsed by sympy
        str_eqn2    equation 2, in a string that can be parsed by sympy

    returns:
        distance   tree edit distance 
        mapping    tree edit mapping (how to get from tree1 to tree2)

    reported in:
        (here)
    """

    sympy_expression_a = sp.simplify(expression_a)
    sympy_expression_b = sp.simplify(expression_b)

    sp_tree_a = sp.srepr(sympy_expression_a)
    sp_tree_b = sp.srepr(sympy_expression_b)

    tree_a = Tree.from_text(tree_to_brackets(sp_tree_a))
    tree_b= Tree.from_text(tree_to_brackets(sp_tree_b))

    ted = APTED(tree_a, tree_b, PerEditOperationConfig(1,1,1))
    distance = ted.compute_edit_distance()
    
    return distance 

def compute_r2_raw(targets, predictions):
    """
    numerical metric
    expression_a is target
    
    """
    
    target_mean = np.mean(targets)

    if np.mean(targets - target_mean) == 0.0:
        eps = 1e-13
    else:
        eps = 0.0
    
    
    ss_residuals = np.sum((targets - predictions)**2)
    ss_total = np.sum((targets - target_mean)**2)

    result =  1.0 - (ss_residuals + eps) / (ss_total + eps)

    result = result.real

    return result

def compute_r2(targets, predictions):
    
    
    r2_raw = compute_r2_raw(targets.reshape(-1,1), predictions.reshape(-1,1))

    return r2_raw

def compute_r2_truncated(targets, predictions):
    """
    numerical metric
    
    reported in:
        Kamienny _et al._ 2022
    """

    r2_raw = compute_r2_raw(targets, predictions)
    # truncate to 0.0 
    r2_truncated = np.clip(r2_raw, 0.0, 1.0)

    return np.mean(r2_truncated)
    

def compute_relative_error(targets, predictions):
    """
    numerical metric

    relative absolute error

    error is calculated relative to targets; this function is not symmetric
    i.e. compute_relative_error(a,b) != compute_relative_error(b,a)

    reported in:
        Vastl _et al._ 2022
    """

    relative_absolute_error = np.mean(np.abs(targets - predictions) / np.abs(targets))

    return relative_absolute_error

def compute_relative_squared_error(targets, predictions):
    """
    numerical metric

    relative absolute error

    error is calculated relative to targets; this function is not symmetric
    i.e. compute_relative_error(a,b) != compute_relative_error(b,a)

    reported in:
        Vastl _et al._ 2022
    """

    relative_squared_error = np.mean(np.abs(targets - predictions)**2 / np.abs(targets)**2)

    return relative_squared_error

def compute_isclose_accuracy(targets, predictions, atol=0.001, rtol=0.05, threshold=0.95): 
    """
    numerical metric (ad-hoc accuracy proxy)

    reported in:
        Biggio _et al._ 2021
        Kamienny _et al._ 2022 (kind of)
    """

    assert targets.shape == predictions.shape, f"expected same shapes {targets.shape}!={predictions.shape}"

    is_close = np.isclose(targets, predictions, atol=atol, rtol=rtol)

    is_close_mean = np.mean(is_close)

    return (is_close_mean >= threshold) 


def compute_r2_over_threshold(targets, predictions, threshold=0.99):
    """
    numerical metric (ad-hoc accuracy proxy)

    reported in:
        Kamienny _et al._ 2022 (with threshold 0.99) (default threshold)
        Biggio _et al._ 2019 (with threshold 0.95)
    """
    
    r2 = compute_r2(targets, predictions)

    return r2 > threshold

def get_r2_threshold_function(threshold):

    def compute_r2_threshold_function(targets, predictions):

        return compute_r2_over_threshold(\
                targets, predictions, threshold=threshold)

    return compute_r2_threshold_function

def get_loss_function(skeleton, y_target, \
        compute_loss=compute_relative_squared_error, **kwargs):

    variables = ", ".join([key for key in kwargs.keys()])
    my_inputs = kwargs


    def loss_function(constants):

        expression = skeleton.replace('C','{}').format(*constants)

        expression_function = sp.lambdify(variables, expr=expression)

        predictions = expression_function(**my_inputs)

        loss = compute_loss(y_target, predictions)

        return loss

    return loss_function

