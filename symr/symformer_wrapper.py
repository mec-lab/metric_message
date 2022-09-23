import os

import time

import numpy as np
import torch
import torch.nn.functional as F

from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper

class SymformerWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
        else:
            self.use_bfgs = False

        self.my_device = torch.device("cpu")
        self.my_beam_width = kwargs["beam_width"] \
                if "beam_width" in kwargs.keys() else 4
        self.initialize_model()
        self.load_parameters()

    def parse_filter(self, expression, variables=["x"]):
        """
        Sometimes cleaning the string isn't enough to yield an actual expression.

        If the expression string is malformed, return f(x) = 0.0 instead
        """
        try: 
            # SymGPT currently only handles one variable: x1
            my_fn = sp.lambdify(variables, expr=expression)
            my_inputs = {key:np.random.rand(3,1) for key in variables}
            _ = my_fn(**my_inputs)

            # return expression if it was successfuly parsed into a function 
            return expression, False

        except:
            return "+".join([f"0.0 * {my_var}" \
                    for my_var in variables]), True

    def __call__(self, target, **kwargs):
        
        t0 = time.time()

        # symformer call goes here
        info = {"failed": True}

        t1 = time.time()

        info["time_elapsed"] = t1-t0


        expression = "+".join([f"0.0 * {my_var}" \
                for my_var in kwargs.keys()])
        
        return expression, info

    def initialize_model(self):
        pass
    def load_parameters(self):
        pass 
