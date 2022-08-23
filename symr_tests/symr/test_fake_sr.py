import os

import unittest

import numpy as np
import sympy as sp


from symr.fake_sr import RandomSR, PolySR, FourierSR, loss_function

"""
class TestComputeIsCloseAccuracy(unittest.TestCase):

    def setUp(self):
        pass
"""
class TestPolySR(unittest.TestCase):

    def setUp(self):
        pass

class TestFourierSR(unittest.TestCase):

    def setUp(self):
        pass

class TestRandomSR(unittest.TestCase):

    def setUp(self):
        pass

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

        loss_aa = loss_function(constants_a, expression, x_input, y_target) 
        loss_ab = loss_function(constants_b, expression, x_input, y_target) 

        self.assertGreater(loss_ab, loss_aa)
        self.assertEqual(loss_aa, 0.0)


if __name__ == "__main__":

    unittest.main(verbosity=2)
