import unittest

import numpy as np

from hopfieldnet.net import HopfieldNetwork
from hopfieldnet.trainers import hebbian_training, calculate_weight, calculate_neuron_weights 


class HebbianTrainingTest(unittest.TestCase):
    def setUp(self):
        self.input_patterns = np.array([[1, -1, 1],
                                        [-1, 1, -1]])

        self.net = HopfieldNetwork(3)

    def test_calculate_weight(self):
        w = calculate_weight(0, 1, self.input_patterns)

        expected_weight = -1.0

        self.assertAlmostEqual(w, expected_weight)

        w = calculate_weight(1, 2, self.input_patterns)

        expected_weight = -1.0

        self.assertAlmostEqual(w, expected_weight)

        w = calculate_weight(0, 2, self.input_patterns)

        expected_weight = 1.0

        self.assertAlmostEqual(w, expected_weight)


    def test_calculate_neuron_weights(self):
        w = calculate_neuron_weights(0, self.input_patterns)

        expected_weights = np.array([0.0, -1.0, 1.0])

        self.assertTrue(np.array_equal(w, expected_weights))

    def test_hebbian_training(self):
        hebbian_training(self.net, self.input_patterns)

        expected_weights = np.array([[0, -1, 1],
                                     [-1, 0, -1],
                                     [1, -1, 0]])

        weights = self.net.get_weights()

        self.assertTrue(np.array_equal(weights, expected_weights), "Test weights not equal")

if __name__ == "__main__":
    unittest.main()
