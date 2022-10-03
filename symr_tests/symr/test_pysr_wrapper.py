import os

import unittest

import numpy as np
import sympy as sp


from symr.wrappers import BaseWrapper
from symr.pysr_wrapper import PySRWrapper

class TestNSRTSWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def test_instantiate(self):

        pysr = PySRWrapper(use_bfgs=True)
        my_inputs = {"x_1": np.arange(0,1.0,0.1)}
        y = np.array(my_inputs["x_1"]**2)

        expression, info = pysr(target=y, **my_inputs)

        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify("x_1", expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))

    def test_call_without_bfgs(self):

        pysr = PySRWrapper(use_bfgs=True)

        my_inputs = {"x_1": np.arange(0,1.0,0.1)}
        y = np.array(my_inputs["x_1"]**2)

        expression, info = pysr(target=y, **my_inputs)

        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify("x_1", expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
                
    def test_pysr_multi(self):

        pysr = PySRWrapper(use_bfgs=True)

        my_inputs = {"x_1": np.arange(0,1.0,0.1),\
                "x_2": np.arange(0,1.0,0.1),\
                "x_3": np.arange(0,1.0,0.1),\
                }
        y = np.array(my_inputs["x_1"]**2+ np.sin(my_inputs["x_2"]) + my_inputs["x_3"])

        expression, info = pysr(target=y, **my_inputs)

        my_vars = ",".join([key for key in my_inputs.keys()])
        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
        self.assertTrue(True)

        my_inputs = {"x_1": np.arange(0,1.0,0.1),\
                "x_2": np.arange(0,1.0,0.1),\
                }

        expression, info = pysr(target=y, **my_inputs)
        my_vars = ",".join([key for key in my_inputs.keys()])

        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))


if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
