"""Tests for the leaderboard-score multiplicative parameter a."""

import math
import unittest

from metakaggle._data_clean import state_transformation


class StateTransformationMultiplierTests(unittest.TestCase):
	def test_default_preserves_a_equals_one_behavior(self) -> None:
		score = 0.8
		expected = math.log(score / (1.0 - score))
		self.assertAlmostEqual(state_transformation(score), expected)

	def test_alternative_a_values_rescale_transformed_score(self) -> None:
		score = 0.8
		baseline = state_transformation(score, a=1.0)

		for a in (0.5, 1.0, 2.0):
			with self.subTest(a=a):
				self.assertAlmostEqual(
					state_transformation(score, a=a),
					a * baseline,
				)

	def test_a_rescales_after_score_clipping(self) -> None:
		upper_bound = math.log(0.9999 / (1.0 - 0.9999))
		self.assertAlmostEqual(
			state_transformation(1.0, a=0.5),
			0.5 * upper_bound,
		)

	def test_a_must_be_positive(self) -> None:
		for a in (0.0, -1.0):
			with self.subTest(a=a):
				with self.assertRaisesRegex(ValueError, "a must be positive"):
					state_transformation(0.8, a=a)


if __name__ == "__main__":
	unittest.main()
