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

from symr_tests.symr.test_wrappers import TestBaseWrapper
from symr_tests.symr.test_nsrts_wrapper import TestNSRTSWrapper
from symr_tests.symr.test_symgpt_wrapper import TestSymGPTWrapper
from symr_tests.symr.test_symformer_wrapper import TestSymformerWrapper
from symr_tests.symr.test_pysr_wrapper import TestPySRWrapper

from symr_tests.symr.test_benchmark import TestEval,\
        TestBenchmark
from symr_tests.symr.test_helpers import TestLoadBenchmark,\
        TestR2OverThreshold,\
        TestR2AUC

if __name__ == "__main__": #pragma: no cover    
    np.random.seed(42)

    unittest.main(verbosity=2)
