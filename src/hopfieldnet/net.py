import numpy as np

class InvalidWeightsException(Exception):
    pass

class InvalidNetworkInputException(Exception):
    pass

class HopfieldNetwork(object):  
    def __init__(self, num_inputs):
        self._num_inputs = num_inputs
        #self._weights = np.zeros((num_inputs,num_inputs))
        self._weights = np.random.uniform(-1.0, 1.0, (num_inputs,num_inputs))
    
    def set_weights(self, weights):
        """Update the weights array"""
        if weights.shape != (self._num_inputs, self._num_inputs):
            raise InvalidWeightsException()
        
        self._weights = weights
    
    def get_weights(self):
        """Return the weights array"""
        return self._weights
    
    def evaluate(self, input_pattern):
        """Calculate the output of the network using the input data"""
        if input_pattern.shape != (self._num_inputs, ):
            raise InvalidNetworkInputException()
        
        sums = input_pattern.dot(self._weights)
        
        s = np.zeros(self._num_inputs)
        
        for i, value in enumerate(sums):
            if value > 0:
                s[i] = 1
            else:
                s[i] = -1
        
        return s 
        
    def run(self, input_pattern, max_iterations=10):
        """Run the network using the input data until the output state doesn't change 
        or a maximum number of iteration has been reached."""
        last_input_pattern = input_pattern
        
        iteration_count = 0
        
        while True:
            result = self.evaluate(last_input_pattern)
            
            iteration_count += 1
            
            if  np.array_equal(result, last_input_pattern) or iteration_count == max_iterations:
                return result
            else:
                last_input_pattern = result