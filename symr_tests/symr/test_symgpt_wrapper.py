import os

import unittest

import numpy as np
import sympy as sp

from symr.wrappers import BaseWrapper
from symr.nsrts_wrapper import NSRTSWrapper
from symr.symgpt_wrapper import SymGPTWrapper
from symr.symformer_wrapper import SymformerWrapper

class TestSymGPTWrapper(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_instantiate(self):
        
        model = SymGPTWrapper() 

        self.assertTrue(True)

    def test_call(self):

        model = SymGPTWrapper() 

        my_inputs = {"z": np.arange(-1,1.0,0.01)}
        y = np.array(my_inputs["z"]**2)

        expression, info = model(target=y, **my_inputs)

        my_vars = ",".join([key for key in my_inputs.keys()])
        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))

    def test_call_w_bfgs(self):

        model = SymGPTWrapper(use_bfgs=True)

        my_inputs = {"x2": np.arange(-1,1.0,0.01)}
        y = np.array(my_inputs["x2"]**2)

        expression, info = model(target=y, **my_inputs)

        my_vars = ",".join([key for key in my_inputs.keys()])
        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))


if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
