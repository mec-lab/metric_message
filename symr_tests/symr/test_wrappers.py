import os

import unittest

import numpy as np
import sympy as sp

from symr.wrappers import BaseWrapper
from symr.nsrts_wrapper import NSRTSWrapper


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

class TestNSRTSWrapper(unittest.TestCase):

    def setUp(self):
        pass

    def test_instantiate(self):

        nsrts = NSRTSWrapper()
        my_inputs = {"x": np.arange(0,1.0,0.1)}
        y = np.array(my_inputs["x"]**2)

        expression = nsrts(target=y, **my_inputs)

        sp_expression = sp.sympify(expression)
        fn_expression = sp.lambdify("x", expression)
        
        _ = fn_expression(**my_inputs)

        self.assertEqual(str, type(expression))
        self.assertTrue(True)
                



if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
