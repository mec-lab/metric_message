import os

import numpy as np

from sympy import lambdify
import sympy as sp
import sklearn
import sklearn.metrics

from scipy.optimize import minimize, least_squares    

from pathlib import Path
from functools import partial
from symr.metrics import get_loss_function

from scipy.optimize import minimize
#
import glob


class PolySR():
    def __init__(self, **kwargs):
        super(PolySR, self).__init__()

        if "input_variables" in kwargs.keys():
            self.variables = kwargs["input_variables"][1:].split(" ")
        else:
            self.variables = ["x"]

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
        else:
            self.use_bfgs = False
        
        self.degree = kwargs["degree"] if "degree" in kwargs.keys() else 5
        self.initialize_model()
        
    def setup_expression(self):
        
        my_polynomial = ""
        
        for variable in self.variables:
            for ii in range(self.degree,0,-1):
                
                my_polynomial += f"C*{variable}**{ii}+"
            
        my_polynomial += "C"
        
        self.expression = my_polynomial

    def initialize_model(self):
        
        self.setup_expression()
        
    def load_parameters(self, filepath=None):
        """
        it doesn't matter if a filepath is provided, this is a fake SR method 
        and has no parameters 
        """
        pass

    def optimize(self, target, **kwargs):

        c = [1.0 for elem in self.expression if elem=="C"]

        loss_function = get_loss_function(self.expression, y_target=target, **kwargs)

        try: 
            optimized = minimize(loss_function, c, method="BFGS")
        except:
            return "+".join([f"0.0 * {my_var}" \
                    for my_var in kwargs.keys()]), {"failed": True}

        optimized_expression = ""
        constants_placed = 0
        for my_char in self.expression:
            if my_char == "C":
                optimized_expression += f"{optimized.x[constants_placed]}"
                constants_placed += 1
            else:
                optimized_expression += my_char

        return optimized_expression, {"failed": False}

    def __call__(self, target=None, **kwargs):
        
        if self.use_bfgs:
            my_expression, info = self.optimize(target=target, **kwargs)
        else:
            my_expression, info = self.expression.replace("C", "1.0"), {"failed": False}
        # so far haven't encounted a failure mode for BFGS on polynomials
        
        return my_expression, info
    
class FourierSR(PolySR):
    def __init__(self, **kwargs):
        super(FourierSR, self).__init__(**kwargs)
                
    def setup_expression(self):
        
        my_fourier = ""
        
        for variable in self.variables:
            for ii in range(self.degree,0,-1):
                
                my_fourier += f"C*sin({variable}*{ii}+C)+"
            
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
        for variable in self.variables:
            for ii in range(self.degree):
                term = np.random.choice(self.terms, p=[1/len(self.terms) for t in self.terms])
                my_term = term.replace("x0", variable)
                self.expression += f"+ {my_term}"
        
    
    def __call__(self, target=None, **kwargs):
        
        # sample a new expression each time.
        self.setup_expression()

        my_expression, info = super().__call__(target=target, **kwargs)

        return my_expression, info
