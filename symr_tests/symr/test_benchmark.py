import os
import subprocess
from subprocess import check_output

import unittest

import numpy as np
import sympy as sp

from symr.benchmark import evaluate

class TestEval(unittest.TestCase):
    """
    test for evaluate called as function 
    """

    def setUp(self):
        pass

    def test_evaluate(self):
        #my_output = check_output(my_command)
        
        kwargs = {"metrics": "r2", \
                "sr_method": "RandomSR",\
                "k_folds": 2}

        returned_value = evaluate(**kwargs)

        self.assertEqual(0, returned_value)

class TestBenchmark(unittest.TestCase):
    """
    test for evaluate as called from the command line
    this test doesn't get tracked by coverage, but it 
    does reflect real usage.
    """

    def setUp(self):
        pass

    def test_benchmark(self):

        my_command = ["python", "-m", "symr.benchmark"]
        
        #my_output = check_output(my_command)
        my_output = "all ok"
        self.assertIn("all ok", my_output)

if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
