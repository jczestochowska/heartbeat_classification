import numpy as np
import os
import tensorflow as tf

from config import PROJECT_ROOT_DIR

TRAIN = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'train.npy')
TEST = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'test.npy')
TRAIN_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                            'train_labels.npy')
TEST_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                           'test_labels.npy')


def main(num_epochs, batch_size=200, training_step=0.2):
    train_features, test_features, train_labels, test_labels = load_dataset()
    random_normal_initializer = tf.initializers.random_normal()

    weights = tf.get_variable('weights', shape=[train_features.shape[1], 1], dtype=tf.float32,
                              initializer=random_normal_initializer)
    bias = tf.get_variable('bias', shape=[1], dtype=tf.float32, initializer=random_normal_initializer)

    features = tf.placeholder(shape=[None, train_features.shape[1]], dtype=tf.float32)
    labels = tf.placeholder(shape=[None], dtype=tf.int64)

    logits = tf.matmul(features, weights) + bias

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(training_step).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for epoch in range(num_epochs):
        batch_features, batch_labels = next_batch(batch_size, train_features, train_labels)
        sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('-------------------------------------------------------------')
        print("Epoch: {}".format(epoch))
        print("training accuracy: ", sess.run(accuracy, feed_dict={features: train_features, labels: train_labels}))
        print("testing accuracy: ", sess.run(accuracy, feed_dict={features: test_features, labels: test_labels}))
        print('-------------------------------------------------------------')


def load_dataset():
    return np.load(TRAIN), np.load(TEST), np.load(TRAIN_LABELS), np.load(TEST_LABELS)


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    return data_shuffle, labels_shuffle


if __name__ == '__main__':
    main(num_epochs=500)
