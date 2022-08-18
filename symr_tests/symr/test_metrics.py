import os

import unittest

import numpy as np
import sympy as sp

from symr.metrics import compute_tree_distance, compute_exact_equivalence

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
    

class TestComputeTreeDistance(unittest.TestCase):
    
    def setUp(self):
        pass

    def test_compute_tree_distance(self):

        expression_a = "x**2 + y**2 + 10"
        expression_b = "x**2 + y**3 + 10"
        expression_c = "x**3 + y**3 + 10"
        expression_d = "sin(x)**2 + 1/y**2 + 10"

        dist_aa = compute_tree_distance(expression_a, expression_b)
        dist_ab = compute_tree_distance(expression_a, expression_b)
        dist_ac = compute_tree_distance(expression_a, expression_b)
        dist_ad = compute_tree_distance(expression_a, expression_b)
        dist_ba = compute_tree_distance(expression_a, expression_b)
        dist_bb = compute_tree_distance(expression_a, expression_b)
        dist_bc = compute_tree_distance(expression_a, expression_b)
        dist_bd = compute_tree_distance(expression_a, expression_b)
        dist_ca = compute_tree_distance(expression_a, expression_b)
        dist_cb = compute_tree_distance(expression_a, expression_b)
        dist_cc = compute_tree_distance(expression_a, expression_b)
        dist_cd = compute_tree_distance(expression_a, expression_b)
        dist_da = compute_tree_distance(expression_a, expression_b)
        dist_db = compute_tree_distance(expression_a, expression_b)
        dist_dc = compute_tree_distance(expression_a, expression_b)
        dist_dd = compute_tree_distance(expression_a, expression_b)

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
