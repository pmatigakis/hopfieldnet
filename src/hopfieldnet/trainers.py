import numpy as np


def calculate_weight(i, j, patterns):
    """Calculate the weight between the given neurons"""
    num_patterns = len(patterns)
    
    s = 0.0
    for mu in range(num_patterns):
        s += patterns[mu][i] * patterns[mu][j]
    
    w = (1.0 / float(num_patterns)) * s

    return w

def calculate_neuron_weights(neuron_index, input_patterns):
    """Calculate the weights for the givven neuron"""
    num_patterns = len(input_patterns)
    num_neurons = len(input_patterns[0])

    weights = np.zeros(num_neurons)

    for j in range(num_neurons):
        if neuron_index == j: continue
        weights[j] = calculate_weight(neuron_index, j, input_patterns)

    return weights

def hebbian_training(network, input_patterns):
    """Train a network using the Hebbian learning rule"""
    n = len(input_patterns)

    num_neurons = len(input_patterns[0])

    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        weights[i] = calculate_neuron_weights(i, input_patterns)

    network.set_weights(weights)
