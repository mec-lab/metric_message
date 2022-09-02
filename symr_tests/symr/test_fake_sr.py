import os

import unittest

import numpy as np
import sympy as sp


from symr.fake_sr import RandomSR, PolySR, FourierSR
from symr.metrics import get_loss_function

"""
class TestComputeIsCloseAccuracy(unittest.TestCase):

    def setUp(self):
        pass
"""
class TestPolySR(unittest.TestCase):

    def setUp(self):
        my_seed = 13
        my_degrees = 10

        np.random.seed(my_seed)
        self.model = PolySR(degree=my_degrees) 

    def test_forward(self):
        expression_a = self.model()
        expression_b = self.model()

        self.assertEqual(expression_a, expression_b)

class TestFourierSR(unittest.TestCase):

    def setUp(self):
        my_seed = 13
        my_degrees = 10

        np.random.seed(my_seed)
        self.model = FourierSR(degree=my_degrees) 

    def test_forward(self):

        expression_a = self.model()
        expression_b = self.model()
        
        self.assertEqual(expression_a, expression_b)

class TestRandomSR(unittest.TestCase):

    def setUp(self):
        
        my_seed = 13
        my_degrees = 10

        np.random.seed(my_seed)
        self.model = RandomSR(degree=my_degrees) 

    def test_forward(self):

        expression_a = self.model()
        expression_b = self.model()

        self.assertNotEqual(expression_a, expression_b)

class TestLossFunction(unittest.TestCase):

    def setUp(self):
        pass

    def test_loss_function(self):

        constants_a = [1.0] * 5
        constants_b = [0.5] * 5


        expression = "C*x0 + x0**C + C - 2*C*x0 + sin(x0*C)"

        expression_a = expression.replace('C','{}').format(*constants_a)

        x_input = np.random.rand(100,1)

        y_target = sp.lambdify("x0", expr=expression_a)(x_input) 

        loss_function = get_loss_function(expression, y_target, x0=x_input)   

        loss_aa = loss_function(constants_a) 
        loss_ab = loss_function(constants_b) 

        self.assertGreater(loss_ab, loss_aa)
        self.assertEqual(loss_aa, 0.0)


if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
