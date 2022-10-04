import os

import unittest

import numpy as np
import sympy as sp

from symr.helpers import load_benchmark, \
        r2_over_threshold, \
        r2_auc


class TestLoadBenchmark(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_load_benchmark(self):
        
        test_benchmark = load_benchmark() 
        test_benchmark_2 = load_benchmark(filepath=None)

        this_filepath = os.path.abspath(__file__)
        root_path = os.path.split(os.path.split(os.path.split(this_filepath)[0])[0])[0]
        bm_filepath = os.path.join("data", "nguyen.csv")

        my_filepath = os.path.join(root_path, bm_filepath)

        nguyen_benchmark = load_benchmark(filepath=my_filepath)

        self.assertEqual(test_benchmark, test_benchmark_2)
        self.assertGreater(len(nguyen_benchmark), \
                len(test_benchmark))

class TestR2OverThreshold(unittest.TestCase):

    def setUp(self):
        pass

    def test_r2_over_threshold(self):
    
        threshold_size = 10

        thresholds = np.arange(0.0, 1.0, 1/threshold_size)

        r2_a = np.random.rand(100,1)
        r2_b = np.ones((100,1))

        r2_ot_a = r2_over_threshold(r2_a, thresholds = thresholds)
        r2_ot_b = r2_over_threshold(r2_b, thresholds = thresholds)
        r2_ot_c = r2_over_threshold(-r2_a, thresholds = thresholds)
        r2_ot_d = r2_over_threshold(-10*r2_a, thresholds = thresholds)

        self.assertEqual(0.0, np.sum(r2_ot_c))
        self.assertEqual(np.sum(r2_ot_d), np.sum(r2_ot_c))

        self.assertEqual(1.0, np.sum(r2_ot_b) / threshold_size)
        self.assertEqual(1.0, r2_ot_a[0])
        self.assertGreater(1.0, r2_ot_a[-1])


class TestR2AUC(unittest.TestCase):
    
    def setUp(self):
        pass


    def test_r2_auc(self):
    
        threshold_size = 10

        thresholds = np.arange(0.0, 1.0, 1/threshold_size)
        thresholds_b = np.arange(0.0, 1.0, 1/threshold_size)

        r2_a = np.random.rand(100,1)
        r2_b = np.ones((100,1))

        r2_ot_a = r2_over_threshold(r2_a, thresholds = thresholds)
        r2_ot_b = r2_over_threshold(r2_a, thresholds = thresholds_b)
        r2_ot_c = r2_over_threshold(-r2_a, thresholds = thresholds_b)

        r2_auc_a = r2_auc(r2_ot_a)
        r2_auc_b = r2_auc(r2_ot_b)
        r2_auc_c = r2_auc(r2_ot_c)

        self.assertEqual(r2_auc_a, r2_auc_b)
        self.assertGreaterEqual(1.0, r2_auc_a)
        self.assertEqual(0.0, r2_auc_c)

if __name__ == "__main__":

    unittest.main(verbosity=2)


