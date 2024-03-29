import os

import unittest

import numpy as np
import sympy as sp

from symr.metrics import compute_tree_distance, compute_exact_equivalence,\
        compute_r2_raw, compute_r2, compute_r2_truncated, \
        compute_isclose_accuracy, compute_r2_over_threshold,\
        get_r2_threshold_function,\
        compute_relative_error, compute_shannon_diversity, \
        compute_complexity, compute_tree_traversal

import sklearn
import sklearn.metrics

"""
class TestComputeIsCloseAccuracy(unittest.TestCase):

    def setUp(self):
        pass
"""

class TestComputeShannonDiversity(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_shannon_diversity(self):

        expression_a = "x"
        expression_b = "x+y+z"
        expression_c = "x*y/z"

        h_a = compute_shannon_diversity(expression_a)
        h_b = compute_shannon_diversity(expression_b)
        h_c = compute_shannon_diversity(expression_c)
        h_c2 = compute_shannon_diversity(expression_c)

        self.assertGreater(h_b, h_a)
        self.assertGreater(h_c, h_a)
        self.assertEqual(h_c, h_c2)

class TestComputeComplexity(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_complexity(self):

        expression_a = "x"
        expression_b = "x+y+z"
        expression_c = "x*y/z"

        h_a = compute_complexity(expression_a)
        h_b = compute_complexity(expression_b)
        h_c = compute_complexity(expression_c)
        h_c2 = compute_complexity(sp.sympify(expression_c))

        self.assertGreater(h_b, h_a)
        self.assertGreater(h_c, h_a)
        self.assertEqual(h_c, h_c2)

class TestComputeTreeTraversal(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_tree_traversal(self):

        expression_a = "x"
        expression_b = "x+y+z"
        expression_c = "x*y/z"

        h_a = compute_complexity(expression_a)
        h_b = compute_complexity(expression_b)
        h_c = compute_complexity(expression_c)

        self.assertGreater(h_b, h_a)
        self.assertGreater(h_c, h_a)

class TestExactEquivalence(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_exact_equivalence(self):

        expression_a = "x**2 + y**2 + 10"
        expression_b = "x**2 + y**3 + 10"

        exact_ab = compute_exact_equivalence(expression_a, expression_b)
        exact_aa = compute_exact_equivalence(expression_a, expression_a)
        exact_bb = compute_exact_equivalence(expression_b, expression_b)

        self.assertTrue(exact_aa)
        self.assertTrue(exact_bb)
        self.assertFalse(exact_ab)
    
class TestComputeRelativeError(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_relative_error(self):

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))
        temp_d = 0.5 * np.ones((32,64))
        
        re_ab = compute_relative_error(temp_a, temp_b)
        re_ba = compute_relative_error(temp_b, temp_a)

        re_cc = compute_relative_error(temp_c, temp_c)
        re_cd = compute_relative_error(temp_c, temp_d)

        self.assertNotEqual(re_ab, re_ba)
        self.assertEqual(re_cc, 0.0)
        self.assertEqual(re_cd, 0.5)

class TestComputeR2(unittest.TestCase):


    def setUp(self):
        my_seed = 13
        np.random.seed(my_seed)

    def test_compute_r2_check(self):
        temp_a = np.random.randn(32,1)
        temp_b = np.random.randn(32,1)

        r2_ab = compute_r2(temp_a, temp_b)
        r2_aa = compute_r2(temp_a, temp_a)

        check_ab = sklearn.metrics.r2_score(temp_a, temp_b)
        check_aa = sklearn.metrics.r2_score(temp_a, temp_a)

        self.assertEqual(r2_ab, check_ab)
        self.assertEqual(r2_aa, check_aa)

        temp_a = np.random.randn(32,100)
        temp_b = np.random.randn(32,100)

        r2_ab = compute_r2(temp_a, temp_b)
        r2_aa = compute_r2(temp_a, temp_a)

        check_ab = sklearn.metrics.r2_score(\
                temp_a.ravel(), temp_b.ravel())
        check_aa = sklearn.metrics.r2_score(\
                temp_a.ravel(), temp_a.ravel())

        self.assertEqual(r2_ab, check_ab)
        self.assertEqual(r2_aa, check_aa)
        
    def test_compute_r2_raw(self):

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))

        r2_ab = compute_r2_raw(temp_a, temp_b)
        r2_aa = compute_r2_raw(temp_a, temp_a)
        r2_cc = compute_r2_raw(temp_c, temp_c)
        
        mean_r2_ab = np.mean(r2_ab)
        mean_r2_aa = np.mean(r2_aa)
        mean_r2_cc = np.mean(r2_cc)

        self.assertEqual(mean_r2_aa, 1.0)
        self.assertLess(mean_r2_ab, mean_r2_aa)
        self.assertAlmostEqual(mean_r2_cc, 0.0)

    def test_compute_r2(self):

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))

        mean_r2_ab = compute_r2(temp_a, temp_b)
        mean_r2_aa = compute_r2(temp_a, temp_a)
        mean_r2_cc = compute_r2(temp_c, temp_c)
        
        self.assertEqual(mean_r2_aa, 1.0)
        self.assertLess(mean_r2_ab, mean_r2_aa)
        self.assertAlmostEqual(mean_r2_cc, 0.0)

    def test_compute_r2_truncated(self):

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))

        mean_r2_ab = compute_r2_truncated(temp_a, temp_b)
        mean_r2_a_anti_a = compute_r2_truncated(temp_a, -temp_a)
        mean_r2_aa = compute_r2_truncated(temp_a, temp_a)
        mean_r2_cc = compute_r2_truncated(temp_c, temp_c)
        
        self.assertEqual(mean_r2_aa, 1.0)
        self.assertLess(mean_r2_ab, mean_r2_aa)
        self.assertAlmostEqual(mean_r2_cc, 0.0)
        self.assertGreaterEqual(mean_r2_a_anti_a, 0.0)

    def test_compute_r2_over_threshold(self): 

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))

        r2_threshold_ab = compute_r2_over_threshold(temp_a, temp_b)
        r2_threshold_aa = compute_r2_over_threshold(temp_a, temp_a)

        r2_threshold_cc = compute_r2_over_threshold(temp_c, temp_c, threshold=-0.1)

        self.assertTrue(r2_threshold_aa)
        self.assertFalse(r2_threshold_ab)
        self.assertTrue(r2_threshold_cc) 

    def test_get_compute_r2_over_threhsold(self):

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)

        temp_c = np.ones((32,64))
        my_compute_r2_over_threshold = get_r2_threshold_function(threshold=-0.1)

        r2_threshold_ab = my_compute_r2_over_threshold(temp_a, temp_b)
        r2_threshold_aa = my_compute_r2_over_threshold(temp_a, temp_a)

        r2_threshold_cc = my_compute_r2_over_threshold(temp_c, temp_c)

        self.assertTrue(r2_threshold_aa)
        self.assertFalse(r2_threshold_ab)
        self.assertTrue(r2_threshold_cc) 
        

class TestComputeIsCloseAccuracy(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_isclose_accuracy(self):

        np.random.seed(13)

        temp_a = np.random.randn(32,64)
        temp_b = np.random.randn(32,64)
        temp_c = np.ones((32,64))
        temp_d = np.ones((20,1))
        temp_dd = 1.0 * temp_d
        temp_dd[:1] = 0.0

        isclose_ab = compute_isclose_accuracy(temp_a, temp_b)
        isclose_aa = compute_isclose_accuracy(temp_a, temp_a)

        isclose_c_smaller_c_true = compute_isclose_accuracy(temp_c, 1.0499 * temp_c, atol=0.00001, rtol=0.05)
        isclose_c_smaller_c_false = compute_isclose_accuracy(temp_c, 1.0499 * temp_c, atol=0.00001, rtol=0.005)

        isclose_ddd_true = compute_isclose_accuracy(temp_d, temp_dd)
        isclose_ddd_false = compute_isclose_accuracy(temp_d, temp_dd, threshold=0.999)

        self.assertFalse(isclose_ab)
        self.assertTrue(isclose_aa)

        self.assertTrue(isclose_c_smaller_c_true)
        self.assertFalse(isclose_c_smaller_c_false)

        self.assertTrue(isclose_ddd_true)
        self.assertFalse(isclose_ddd_false)

        
class TestComputeTreeDistance(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_compute_tree_distance(self):

        expression_a = "x**2 + y**2 + 10"
        expression_b = "x**2 + y**3 + 10"
        expression_c = "x**3 + y**3 + 10"
        expression_d = "sin(x)**2 + 1/y**2 + 10"

        dist_aa = compute_tree_distance(expression_a, expression_a)
        dist_ab = compute_tree_distance(expression_a, expression_b)
        dist_ac = compute_tree_distance(expression_a, expression_c)
        dist_ad = compute_tree_distance(expression_a, expression_d)

        dist_ba = compute_tree_distance(expression_b, expression_a)
        dist_bb = compute_tree_distance(expression_b, expression_b)
        dist_bc = compute_tree_distance(expression_b, expression_c)
        dist_bd = compute_tree_distance(expression_b, expression_d)

        dist_ca = compute_tree_distance(expression_c, expression_a)
        dist_cb = compute_tree_distance(expression_c, expression_b)
        dist_cc = compute_tree_distance(expression_c, expression_c)
        dist_cd = compute_tree_distance(expression_c, expression_d)

        dist_da = compute_tree_distance(expression_d, expression_a)
        dist_db = compute_tree_distance(expression_d, expression_b)
        dist_dc = compute_tree_distance(expression_d, expression_c)
        dist_dd = compute_tree_distance(expression_d, expression_d)

        self.assertEqual(dist_aa, 0)
        self.assertEqual(dist_bb, 0)
        self.assertEqual(dist_cc, 0)
        self.assertEqual(dist_dd, 0)

        self.assertEqual(dist_ab, dist_ba)
        self.assertEqual(dist_ac, dist_ca)
        self.assertEqual(dist_ad, dist_da)

        self.assertEqual(dist_bc, dist_cb)
        self.assertEqual(dist_bd, dist_db)

        self.assertEqual(dist_cd, dist_dc)

        self.assertEqual(dist_ab, 1)
        self.assertEqual(dist_ac, 2)


if __name__ == "__main__": #pragma: no cover

    unittest.main(verbosity=2)
