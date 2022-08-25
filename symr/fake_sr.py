import os
import json 
import time

import numpy as np

from sympy import lambdify
import sympy as sp
import sklearn
import sklearn.metrics

from scipy.optimize import minimize, least_squares    

from pathlib import Path
from functools import partial

#
import glob

def loss_function(constants, skeleton, x_input, y_target):
    
    equation = skeleton.replace('C','{}').format(*constants)
    
    # this way is faster
    return np.mean((y_target-sp.lambdify("x0", expr=equation)(x_input))**2) 

class PolySR():
    def __init__(self, **kwargs):
        super(PolySR, self).__init__()
        
        self.degree = kwargs["degree"] if "degree" in kwargs.keys() else 5
        self.setup_expression()
        
    def setup_expression(self):
        
        my_polynomial = ""
        
        for ii in range(self.degree,0,-1):
            
            my_polynomial += f"C*x0**{ii}+"
            
        my_polynomial += "C"
        
        self.expression = my_polynomial
        
    def __call__(self, x=None, y=None):
        
        return self.expression
    
class FourierSR(PolySR):
    def __init__(self, **kwargs):
        super(FourierSR, self).__init__(**kwargs)
                
    def setup_expression(self):
        
        my_fourier = ""
        
        for ii in range(self.degree,0,-1):
            
            my_fourier += f"C*sin(x0*{ii}+C)+"
            
        my_fourier += "C"
        
        self.expression = my_fourier
        
class RandomSR(PolySR):
    
    def __init__(self, **kwargs):
        
        
        self.terms = ["C*sin(x0*C+C)",\
                      "C*cos(x0*C+C)",\
                      "C*tan(x0*C+C)",\
                      "C*x0**C", "C", \
                      "C*exp(C*x0)",\
                      "C*x0", \
                      "C*x0**2", \
                      "C*x0**3", \
                      "C*x0**4", \
                      "C*x0**5", \
                      "C*x0**6",\
                      "sqrt(abs(x0))"]

        super(RandomSR, self).__init__(**kwargs)
        
    def setup_expression(self):
        
        # sample expression
        self.expression = "C"
        for ii in range(self.degree):
            term = np.random.choice(self.terms, p=[1/len(self.terms) for t in self.terms])
            self.expression += f"+ {term}"
        
    
    def __call__(self, x=None, y=None):
        
        # sample a new expression each time.
        self.setup_expression()

        self.expression = str(sp.simplify(self.expression)) 

        return self.expression
