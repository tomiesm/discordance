import sys
from pathlib import Path
import unittest
import itertools
import numpy as np
from src.paper.cells.emt_spatial_enrichment_v1.spatial_test import graph, null_moments, permuted_marks, split_groups, holm, pvalue


class StatisticalChecks(unittest.TestCase):
    def test_moments_and_pairs_equal_exhaustive_stratified_null(self):
        coords = np.array([[0,0], [1,0], [2,0], [0,1], [1,1], [3,1]], float)
        a, edges = graph(coords, 1.5)
        groups = [np.array([0,1,2]), np.array([3,4,5])]
        y = np.array([1,0,0,1,1,0], float)
        mu, var, pair_mean, _, _ = null_moments(a, groups, y)
        neighbor_values = []
        pair_values = []
        for first in itertools.combinations(groups[0], 1):
            for second in itertools.combinations(groups[1], 2):
                draw = np.zeros(6)
                draw[list(first)+list(second)] = 1
                neighbor_values.append(a@draw)
                pair_values.append(sum(draw[i]*draw[j] for i,j in edges))
        np.testing.assert_allclose(mu, np.mean(neighbor_values, axis=0), atol=1e-12)
        np.testing.assert_allclose(var, np.var(neighbor_values, axis=0), atol=1e-12)
        self.assertAlmostEqual(pair_mean, np.mean(pair_values))

    def test_permutations_preserve_all_stratum_counts(self):
        groups = [np.arange(4), np.arange(4,9), np.arange(9,12)]
        draws = permuted_marks(groups, [2,0,3], 12, 100, np.random.default_rng(4))
        for ix, k in zip(groups, [2,0,3]):
            np.testing.assert_array_equal(draws[ix].sum(axis=0), np.full(100, k))

    def test_fixed_locations_respect_hole_and_distance(self):
        a, edges = graph(np.array([[0,0],[1,0],[10,0],[11,0]], float), 2)
        self.assertEqual(set(map(tuple, edges)), {(0,1),(2,3)})
        np.testing.assert_array_equal(a.diagonal(), np.ones(4))

    def test_quantile_split_does_not_split_ties_or_tiny_groups(self):
        ix = np.arange(90)
        groups = split_groups([ix], np.r_[np.zeros(85),np.ones(5)], 3)
        self.assertEqual(len(groups), 1)
        groups = split_groups([np.arange(20)], np.arange(20), 3)
        self.assertEqual(len(groups), 1)

    def test_monte_carlo_pvalue_not_zero_and_ties_included(self):
        self.assertEqual(pvalue(np.array([1,2,3]), 4), .25)
        self.assertEqual(pvalue(np.array([1,2,3]), 3), .5)

    def test_holm_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.04,.001,.03]), [.06,.003,.06])


if __name__ == '__main__':
    unittest.main()
