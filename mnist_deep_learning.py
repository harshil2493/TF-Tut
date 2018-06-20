import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
input > weights > hidden l1 (activation) > weights > hidden l2 (activation) > weights > output

compare output with intended output > cost or loss function

Optimization function optimizer > minimizecost (SGD, AdaGrad, AdamOptimizer)

Manipulation backpropagation

feed forward + backprop == epoch 
'''

mnist = input_data.read_data_sets("/tmp/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10

batch_size = 100

features = 784
x = tf.placeholder('float', [None, features])
y = tf.placeholder('float')

weights_key = 'weights'
biases_key = 'biases'

print("mnist.test.images: ", mnist.test.images)
print("mnist.train.images: ", mnist.train.images)
print("mnist.test.labels: ", mnist.test.labels)
print("mnist.train.labels: ", mnist.train.labels)


def neural_network_model(data):
    # (input data weights) + biases
    hidden_1_layer = {weights_key: tf.Variable(tf.random_normal([features, n_nodes_hl1])),
                      biases_key: tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {weights_key: tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      biases_key: tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {weights_key: tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      biases_key: tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {weights_key: tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    biases_key: tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer[weights_key]),
                hidden_1_layer[biases_key])

    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer[weights_key]),
                hidden_2_layer[biases_key])

    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer[weights_key]),
                hidden_3_layer[biases_key])

    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer[weights_key]),
                    output_layer[biases_key])

    return output


def train_neural_network(x):
    prediction = neural_network_model(data=x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycle forward backward
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print("Epoch: ", epoch, " Loss: ", epoch_loss)

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x=x)
