"""Scientific regression tests for the missing expression-baseline correction."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.regressors import FixedAlphaRidgeRegressor


class RidgeCalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(42)
        self.x = rng.normal(size=(240, 12))
        self.test_x = rng.normal(size=(45, 12))
        self.y = self.x @ rng.normal(scale=0.15, size=(12, 3)) + [2.0, 4.0, 7.0]

    def test_constant_expression_is_recovered(self):
        y = np.broadcast_to([2.0, 4.0, 7.0], (len(self.x), 3)).copy()
        reg = FixedAlphaRidgeRegressor(pca_components=8)
        reg.fit(self.x, y)
        assert_allclose(reg.predict(self.test_x), np.broadcast_to(y[0], (45, 3)), atol=1e-10)

    def test_expression_translation_shifts_predictions_not_residuals(self):
        shift = np.array([1.5, 3.0, 8.0])
        original = FixedAlphaRidgeRegressor(pca_components=8)
        translated = FixedAlphaRidgeRegressor(pca_components=8)
        original.fit(self.x, self.y)
        translated.fit(self.x, self.y + shift)
        assert_allclose(translated.predict(self.test_x), original.predict(self.test_x) + shift,
                        atol=1e-9, rtol=1e-9)
        assert_allclose((self.y + shift) - translated.predict(self.x),
                        self.y - original.predict(self.x), atol=1e-9, rtol=1e-9)
        assert_allclose(original.predict(self.x).mean(axis=0), self.y.mean(axis=0), atol=1e-9)

    def test_validation_targets_cannot_change_training_baseline(self):
        first = FixedAlphaRidgeRegressor(pca_components=8)
        second = FixedAlphaRidgeRegressor(pca_components=8)
        first.fit(self.x, self.y, X_val=self.test_x, Y_val=np.zeros((45, 3)))
        second.fit(self.x, self.y, X_val=self.test_x, Y_val=np.full((45, 3), 1000.0))
        assert_allclose(first.predict(self.test_x), second.predict(self.test_x), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
