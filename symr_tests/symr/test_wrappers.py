import os

import unittest

import numpy as np
import sympy as sp

from symr.wrappers import BaseWrapper
from symr.nsrts_wrapper import NSRTSWrapper
from symr.symgpt_wrapper import SymGPTWrapper



class TestBaseWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def test_false(self):

        class FakeWrapper(BaseWrapper): 
            """
            does not include overrides for mandatory classes
            """
            def __init__(self): #pragma: no cover
                # this is intended to fail, and is therefore not included in coverage
                super(FakeWrapper, self).__init__()
        try: 
            wrapper = FakeWrapper()
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_true(self):

        class TempWrapper(BaseWrapper): 
            
            def __init__(self):
                super(TempWrapper, self).__init__()

            def __call__(self):
                return True

            def initialize_model(self):
                return True

            def load_parameters(self):
                return True

        wrapper = TempWrapper()

        self.assertTrue(wrapper.initialize_model())
        self.assertTrue(wrapper())
        self.assertTrue(wrapper.load_parameters())

class TestSymGPTWrapper(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_instantiate(self):
        
        model = SymGPTWrapper() 

class TestNSRTSWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def test_instantiate(self):

        nsrts = NSRTSWrapper(use_bfgs=True)
        my_inputs = {"x_1": np.arange(0,1.0,0.1)}
        y = np.array(my_inputs["x_1"]**2)

        expression = nsrts(target=y, **my_inputs)

        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify("x_1", expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
        self.assertTrue(True)
                
    def test_nsrts_multi(self):

        nsrts = NSRTSWrapper(use_bfgs=True)
        my_inputs = {"x_1": np.arange(0,1.0,0.1),\
                "x_2": np.arange(0,1.0,0.1),\
                "x_3": np.arange(0,1.0,0.1),\
                }
        y = np.array(my_inputs["x_1"]**2+ np.sin(my_inputs["x_2"]) + my_inputs["x_3"])

        expression = nsrts(target=y, **my_inputs)

        my_vars = ",".join([key for key in my_inputs.keys()])
        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
        self.assertTrue(True)

        my_inputs = {"x_1": np.arange(0,1.0,0.1),\
                "x_2": np.arange(0,1.0,0.1),\
                }

        my_vars = ",".join([key for key in my_inputs.keys()])
        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify(my_vars, expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
        self.assertTrue(True)



if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
