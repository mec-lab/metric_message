import os

import time

import numpy as np

from sympy import lambdify
import sympy as sp

import tensorflow as tf
from symformer.model.runner import Runner

from symr.wrappers import BaseWrapper

class SymformerWrapper(BaseWrapper):

    def __init__(self, **kwargs):

        if "use_bfgs" in kwargs.keys():
            self.use_bfgs = kwargs["use_bfgs"]
            self.optimization_type = "bfgs_init"
        else:
            self.use_bfgs = False
            self.optimization_type = "no_opt"

        if "model_name" in kwargs.keys():
            self.model_name = kwargs["model_name"]
        else:
            self.model_name = "symformer-univariate" 

        self.my_device = "cpu"
        self.beam_width = kwargs["beam_width"] \
                if "beam_width" in kwargs.keys() else 4

    
        self.initialize_model()



    def parse_filter(self, expression, support=[[0,1]], variables=["x"]):
        """
        Sometimes cleaning the string isn't enough to yield an actual expression.

        If the expression string is malformed, return f(x) = 1.0*x instead
        """
        try: 
            # SymGPT currently only handles one variable: x1
            my_fn = sp.lambdify(variables, expr=expression)

            my_inputs = {}
            for idx, key in enumerate(variables):
                range_stretcr = (support[idx][1] - support[idx][0])
                my_input = range_stretch * np.random.rand(10,1) - support[idx][0]
                my_inputs[key] = my_input 

            _ = my_fn(**my_inputs)

            # return expression if it was successfuly parsed into a function 
            return expression, False

        except:
            return "+".join([f"1.0 * {my_var}" \
                    for my_var in variables]), True

    def __call__(self, target, **kwargs):
        
        t0 = time.time()

        info = {}
        # symformer call goes here
        try:
            x = kwargs[list(kwargs.keys())[0]].reshape(-1,1)
            if "bivariate" in self.model_name:
                x = np.append(x, \
                        kwargs[list(kwargs.keys())[1]].reshape(-1,1),\
                        axis=-1)

            points = np.append(x, \
                    target.reshape(-1,1), axis=-1)

            points = tf.convert_to_tensor(points[None,:,:])

            result = self.runner.predict(\
                    equation="x", points=points)

            expression = result[0]

            info = {"failed": False}
        except:
            expression = "+".join([f"1.0 * {my_var}" \
                    for my_var in kwargs.keys()])

            info = {"failed": True}
            t1 = time.time()
            info["time_elapsed"] = t1-t0

            return expression, info

        my_vars = ["x", "y"]
        for idx, key in enumerate(kwargs.keys()):
            if idx > 1:
                break
            expression = expression.replace("exp", "@@@")
            expression = expression.replace(f"{my_vars[idx]}", key)
            expression = expression.replace("@@@", "exp")


        support = []
        for key in kwargs.keys():
            support.append([np.min(kwargs[key]), np.max(kwargs[key])])

        expression, info["failed"] = self.parse_filter(expression, support, kwargs.keys())

        t1 = time.time()

        info["time_elapsed"] = t1-t0
        
        return expression, info

    def initialize_model(self):

        self.load_parameters(self.model_name)

    def load_parameters(self, model_name):

        self.runner = Runner.from_checkpoint(\
                self.model_name, \
                num_equations=self.beam_width,\
                optimization_type=self.optimization_type)

