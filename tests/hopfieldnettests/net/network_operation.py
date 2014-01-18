import unittest

import numpy as np

from hopfieldnet.net import HopfieldNetwork, InvalidNetworkInputException

class HopfieldNetworkOperationTests(unittest.TestCase):
    def setUp(self):
        self.net = HopfieldNetwork(3)
    
    def test_evaluate(self):
        pattern = np.array([-1, 1, -1])
        output = np.array([1, -1, 1])
        
        weights = np.array([[0, 1, -1],
                            [1, 0, 0],
                            [-1, 1, 0]])
        
        self.net.set_weights(weights)
        
        result = self.net.evaluate(pattern)
        
        self.assertTrue(np.array_equal(result, output), "The network outputs doesn't match the expected result")

    def test_throw_exception_on_invalid_input(self):
        input_pattern = np.ones(10)
        
        self.assertRaises(InvalidNetworkInputException, self.net.evaluate, input_pattern)

if __name__ == "__main__":
    unittest.main()