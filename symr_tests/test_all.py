import os

import unittest

import numpy as np
import sympy as sp

from symr_tests.symr.test_metrics import TestExactEquivalence,\
        TestComputeTreeDistance,\
        TestComputeR2,\
        TestComputeIsCloseAccuracy,\
        TestComputeRelativeError,\
        TestComputeShannonDiversity,\
        TestComputeComplexity

from symr_tests.symr.test_fake_sr import TestPolySR,\
        TestFourierSR,\
        TestRandomSR,\
        TestLossFunction 

from symr_tests.symr.test_wrappers import TestBaseWrapper,\
        TestNSRTSWrapper
from symr_tests.symr.test_benchmark import TestEval,\
        TestBenchmark

if __name__ == "__main__": #pragma: no cover    

    unittest.main(verbosity=2)
