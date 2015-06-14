import unittest

import numpy as np

from hopfieldnet.net import HopfieldNetwork, InvalidNetworkInputException

class HopfieldNetworkOperationTests(unittest.TestCase):
    def setUp(self):
        self.net = HopfieldNetwork(3)

        self.input_patterns = np.array([[1, -1, 1],
                                        [-1, 1, -1]])

        weights = np.array([[0.0, -1.0, 1.0],
                            [-1.0, 0.0, -1.0],
                            [1.0, -1.0, 0.0]])

        self.net.set_weights(weights)

    def test_calculate_neuron_output(self):
        neuron_output = self.net.calculate_neuron_output(0, self.input_patterns[0])

        expected_neuron_output = 1.0

        self.assertAlmostEqual(neuron_output, expected_neuron_output, 3)

        neuron_output = self.net.calculate_neuron_output(1, self.input_patterns[0])

        expected_neuron_output = -1.0

        self.assertAlmostEqual(neuron_output, expected_neuron_output, 3)

        neuron_output = self.net.calculate_neuron_output(2, self.input_patterns[0])

        expected_neuron_output = 1.0

        self.assertAlmostEqual(neuron_output, expected_neuron_output, 3)

if __name__ == "__main__":
    unittest.main()
