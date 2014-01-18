from random import randint

import numpy as np
from matplotlib import pyplot as plt

from hopfieldnet.net import  HopfieldNetwork
from hopfieldnet.trainers import hebbian_training

#Create the training patterns
a_pattern = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 0, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1]])

b_pattern = np.array([[1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0],
                      [1, 0, 0, 0, 1],
                      [1, 0, 0, 0, 1],
                      [1, 1, 1, 1, 0]])

c_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])

a_pattern *= 2
a_pattern -= 1

b_pattern *= 2
b_pattern -= 1

c_pattern *= 2
c_pattern -= 1

input_patterns = np.array([a_pattern.flatten(), b_pattern.flatten(), c_pattern.flatten()])

#Create the neural network and train it using the training patterns
network = HopfieldNetwork(35)

hebbian_training(network, input_patterns)

#Create the test patterns by using the training patterns and adding some noise to them
#and use the neural network to denoise them 
a_test =  a_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    a_test[p] *= -1
    
a_result = network.run(a_test)

a_result.shape = (7, 5)
a_test.shape = (7, 5)

b_test =  b_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    b_test[p] *= -1
    
b_result = network.run(b_test)

b_result.shape = (7, 5)
b_test.shape = (7, 5)

c_test =  c_pattern.flatten()

for i in range(4):
    p = randint(0, 34)
    c_test[p] *= -1
    
c_result = network.run(c_test)

c_result.shape = (7, 5)
c_test.shape = (7, 5)

#Show the results
plt.subplot(3, 2, 1)
plt.imshow(a_test, interpolation="nearest")
plt.subplot(3, 2, 2)
plt.imshow(a_result, interpolation="nearest")

plt.subplot(3, 2, 3)
plt.imshow(b_test, interpolation="nearest")
plt.subplot(3, 2, 4)
plt.imshow(b_result, interpolation="nearest")

plt.subplot(3, 2, 5)
plt.imshow(c_test, interpolation="nearest")
plt.subplot(3, 2, 6)
plt.imshow(c_result, interpolation="nearest")

plt.show()
