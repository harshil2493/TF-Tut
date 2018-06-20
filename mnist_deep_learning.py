from deep_learning_network import train_neural_network
from tensorflow.examples.tutorials.mnist import input_data

'''
input > weights > hidden l1 (activation) > weights > hidden l2 (activation) > weights > output

compare output with intended output > cost or loss function

Optimization function optimizer > minimizecost (SGD, AdaGrad, AdamOptimizer)

Manipulation backpropagation

feed forward + backprop == epoch 
'''

mnist = input_data.read_data_sets("/tmp/", one_hot=True)

print("mnist.train.images: ", mnist.train.images)
train_x = mnist.train.images

print("mnist.test.images: ", mnist.test.images)
test_x = mnist.test.images

print("mnist.train.labels: ", mnist.train.labels)
train_y = mnist.train.labels

print("mnist.test.labels: ", mnist.test.labels)
test_y = mnist.test.labels

train_neural_network(train_x=train_x,
                     train_y=train_y,
                     test_x=test_x,
                     test_y=test_y)
