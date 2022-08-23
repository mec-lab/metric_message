import os

import unittest

import numpy as np
import sympy as sp

from symr_tests.symr.test_metrics import TestExactEquivalence,\
        TestComputeTreeDistance,\
        TestComputeR2,\
        TestComputeIsCloseAccuracy,\
        TestComputeRelativeError

from symr_tests.symr.test_fake_sr import TestPolySR,\
        TestFourierSR,\
        TestRandomSR,\
        TestLossFunction 

if __name__ == "__main__":

    unittest.main(verbosity=2)
