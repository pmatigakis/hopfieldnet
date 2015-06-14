import numpy as np
from random import randint, shuffle

class InvalidWeightsException(Exception):
    pass


class InvalidNetworkInputException(Exception):
    pass

class HopfieldNetwork(object):
    def __init__(self, num_inputs):
        self._num_inputs = num_inputs
        self._weights = np.random.uniform(-1.0, 1.0, (num_inputs, num_inputs))

    def set_weights(self, weights):
        """Update the weights array"""
        if weights.shape != (self._num_inputs, self._num_inputs):
            raise InvalidWeightsException()

        self._weights = weights

    def get_weights(self):
        """Return the weights array"""
        return self._weights
    
    def calculate_neuron_output(self, neuron, input_pattern):
        """Calculate the output of the given neuron"""
        num_neurons = len(input_pattern)

        s = 0.0

        for j in range(num_neurons):
            s += self._weights[neuron][j] * input_pattern[j]

        return 1.0 if s > 0.0 else -1.0

    def run_once(self, update_list, input_pattern):
        """Iterate over every neuron and update it's output"""
        result = input_pattern.copy()

        changed = False
        for neuron in update_list:
            neuron_output = self.calculate_neuron_output(neuron, result)

            if neuron_output != result[neuron]:
                result[neuron] = neuron_output
                changed = True

        return changed, result

    def run(self, input_pattern, max_iterations=10):
        """Run the network using the input data until the output state doesn't change
        or a maximum number of iteration has been reached."""
        iteration_count = 0

        result = input_pattern.copy()

        while True:
            update_list = range(self._num_inputs)
            shuffle(update_list)

            changed, result = self.run_once(update_list, result)

            iteration_count += 1

            if not changed or iteration_count == max_iterations:
                return result
