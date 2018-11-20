import itertools
import os

import tensorflow as tf

from config import PROJECT_ROOT_DIR
from models.dataset_utils import load_dataset, get_balanced_dataset, SUMMARIES_DIR


def logistic_regression_training(num_epochs=1000, batch_size=20, training_step=0.1):
    train_features, test_features, train_labels, test_labels = load_dataset()

    balanced_dataset = get_balanced_dataset(train_features, train_labels)
    balanced_dataset = balanced_dataset.repeat().batch(batch_size)

    sess = tf.InteractiveSession()
    iterator = balanced_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    labels = tf.cast(labels, tf.int64)

    weights = tf.Variable(tf.zeros([train_features.shape[1], 2], dtype=tf.float64))
    bias = tf.Variable(tf.zeros([2], dtype=tf.float64))

    logits = tf.matmul(features, weights) + bias

    with tf.name_scope('cross_entropy'):
        with tf.name_scope('total'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    with tf.name_scope('optimization'):
        optimizer = tf.train.GradientDescentOptimizer(training_step).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
        with tf.name_scope('precision'):
            precision, precision_update = tf.metrics.precision(labels=labels,
                                                               predictions=tf.argmax(logits, 1),
                                                               name="precision")
        with tf.name_scope('recall'):
            recall, recall_update = tf.metrics.recall(labels=labels,
                                                      predictions=tf.argmax(logits, 1),
                                                      name="recall")
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision_update)
    tf.summary.scalar('recall', recall_update)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(SUMMARIES_DIR, 'logistic_regression',
                                                      'train_epochs{}_bs{}_lr{}'.format(epochs, batch_size,
                                                                                        training_step)),
                                         sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(SUMMARIES_DIR, 'logistic_regression',
                                                     'test_epochs{}_bs{}_lr{}'.format(epochs, batch_size, training_step)))

    sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
        _, loss_value = sess.run([optimizer, cross_entropy])
        if epoch % 10 == 0:
            train_summary = sess.run(merged)
            train_writer.add_summary(train_summary, epoch)
        test_summary = sess.run(merged, feed_dict={features: test_features, labels: test_labels})
        test_writer.add_summary(test_summary, epoch)

    path = os.path.join(PROJECT_ROOT_DIR, "models", "logistic_regression_weights")
    inputs_dict = {
        "features": features,
        "labels": labels
    }
    outputs_dict = {
        "logits": logits
    }
    tf.saved_model.simple_save(
        sess, path, inputs_dict, outputs_dict
    )


if __name__ == '__main__':
    epochs = [1000, 2000, 3000]
    batches = [20, 50, 100]
    training_steps = [0.1, 0.2, 0.01, 0.001]
    hyperparameters = [epochs, batches, training_steps]
    hyperparameters = list(itertools.product(*hyperparameters))
    for hyperparameters_set in hyperparameters:
        logistic_regression_training(hyperparameters_set)
