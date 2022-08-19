import os

import unittest

import numpy as np
import sympy as sp

from symr.metrics import compute_tree_distance, compute_exact_equivalence,\
        compute_r2_raw, compute_r2, compute_r2_truncated, \
        compute_isclose_accuracy

"""
class TestComputeIsCloseAccuracy(unittest.TestCase):

    def setUp(self):
        pass
"""

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
    
class TestComputeR2(unittest.TestCase):

    def setUp(self):
        pass

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

        self.assertTrue(r2_ab.shape == temp_a.shape)
        self.assertTrue(r2_ab.shape == temp_b.shape)
        self.assertTrue(r2_aa.shape == temp_a.shape)
        self.assertTrue(r2_cc.shape == temp_c.shape)


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
        self.assertGreater(mean_r2_a_anti_a, 0.0)

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

if __name__ == "__main__":

    unittest.main(verbosity=2)
