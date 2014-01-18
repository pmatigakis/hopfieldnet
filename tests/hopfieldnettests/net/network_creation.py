import unittest

import numpy as np

from hopfieldnet.net import HopfieldNetwork, InvalidWeightsException

class HopfieldNetworkCreationTests(unittest.TestCase):
    def setUp(self):
        self.net = HopfieldNetwork(10)
    
    def test_change_network_weights(self):
        new_weights = np.ones((10, 10))
        
        self.net.set_weights(new_weights)
        
        self.assertTrue(np.array_equal(self.net.get_weights(), new_weights), "The network weights have not been updated")

    def test_fail_to_change_weights_when_shape_not_same_as_input_vector(self):
        new_weights = np.ones((5, 5))
        
        self.assertRaises(InvalidWeightsException, self.net.set_weights, new_weights)

    def test_network_creation(self):
        self.assertEqual(self.net.get_weights().shape, (10, 10), "The networks weight array has wrong shape")

if __name__ == "__main__":
    unittest.main()