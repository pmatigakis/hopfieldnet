import numpy as np


def hebbian_training(network, input_patterns):
    """Train a network using the Hebbian learning rule"""
    n = len(input_patterns)

    num_neurons = network.get_weights().shape[0]

    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j:
                continue
            for m in range(n):
                weights[i, j] += input_patterns[m][i] * input_patterns[m][j]

    weights *= 1 / float(n)

    network.set_weights(weights)
