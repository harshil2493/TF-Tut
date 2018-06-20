import tensorflow as tf

"""
Creating Network!!
"""


def neural_network_model(data,
                         features, n_classes,
                         n_nodes_hl1=500, n_nodes_hl2=500, n_nodes_hl3=500,
                         weights_key='weights', biases_key='biases'):
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


"""
Training NN
"""


def train_neural_network(train_x, train_y, test_x, test_y, batch_size=1000):
    features = len(train_x[0])
    n_classes = len(train_y[0])

    x = tf.placeholder('float', [None, features])
    y = tf.placeholder('float')

    prediction = neural_network_model(data=x,
                                      features=features,
                                      n_classes=n_classes)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycle forward backward
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                epoch_x = train_x[start:end]
                epoch_y = train_y[start:end]
                o, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

                i += batch_size

            print("Epoch: ", epoch, " Loss: ", epoch_loss)

        correct = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print("Accuracy: ", accuracy.eval({x: test_x, y: test_y}))
