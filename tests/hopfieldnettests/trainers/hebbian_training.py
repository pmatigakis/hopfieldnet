import unittest

import numpy as np

from hopfieldnet.net import HopfieldNetwork
from hopfieldnet.trainers import hebbian_training

class HebbianTrainingTest(unittest.TestCase):
    def setUp(self):
        self.input_patterns = np.array([[1, -1, 1],
                                        [-1, 1, -1]])
    
        self.net = HopfieldNetwork(3)
    
    def test_hebbian_training(self):
        hebbian_training(self.net, self.input_patterns)
        
        expected_weights = np.array([[0, -1, 1],
                                     [-1, 0, -1],
                                     [1, -1, 0]])
        
        weights = self.net.get_weights()
        
        self.assertTrue(np.array_equal(weights, expected_weights), "The hebbian trainer failed to find the correct weights")

if __name__ == "__main__":
    unittest.main()