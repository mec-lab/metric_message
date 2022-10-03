import os
import json 
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
my_cmap = plt.get_cmap("viridis")
from sympy import lambdify
import sympy as sp

from symr.wrappers import BaseWrapper
from pysr import PySRRegressor

class PySRWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
        else:
            self.use_bfgs = False

        self.initialize_model()
        self.load_parameters()


    def parse_filter(self, expression, variables=["x"]):
        """
        Sometimes cleaning the string isn't enough to yield an actual expression.

        If the expression string is malformed, return f(x) = 1.0*x instead
        """
        try: 
            # SymGPT currently only handles one variable: x1
            my_fn = sp.lambdify(variables, expr=expression)
            my_inputs = {key:np.random.rand(3,1) for key in variables}
            _ = my_fn(**my_inputs)

            # return expression if it was successfuly parsed into a function 
            return expression, False

        except:
            return "+".join([f"1.0 * {my_var}" \
                    for my_var in variables]), True

    def no_bfgs_inference(self, x, target):

        best_expression = ""

        return best_expression

    def __call__(self, target, **kwargs):
        
        t0 = time.time()

        keys = list(kwargs.keys())

        x = kwargs[keys[0]].reshape(-1,1)

        for index, key in enumerate(keys[1:]):
            
            x = np.append(x, kwargs[key].reshape(-1,1), axis=-1)

        try:
            self.model.fit(x, target)

            expression = str(self.model.get_best()["sympy_format"])

            for idx, key in enumerate(kwargs.keys()):
                expression = expression.replace(f"x{idx}", key)

            expression, failed = self.parse_filter(expression, kwargs.keys())

            t1 = time.time()
            info = {"failed": failed}
            info["time_elapsed"] = t1-t0

        except:

            expression =  "+".join([f"1.0 * {my_var}" \
                    for my_var in kwargs.keys()])

            t1 = time.time()
            info = {"failed": True}
            info["time_elapsed"] = t1-t0


        return expression, info

    def initialize_model(self):

        # TODO: allow different settings here

        self.model = PySRRegressor(
            niterations=10,
            binary_operators=["+", "*"],
            unary_operators=[
                "cos",
                "exp",
                "sin",
                "sqrt",
                "inv(x) = 1/x"  # Custom operator (julia syntax)
            ],
            model_selection="best",
            deterministic = True,
            procs = 0,
            multithreading = False,
            random_state = 42,
            verbosity=0,
            loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
        )
        self.model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})

    def load_parameters(self, filepath=None):

        pass

