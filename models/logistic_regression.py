import tensorflow as tf

from models.dataset_utils import load_dataset, get_balanced_dataset, delete_tensorboard_summaries, SUMMARIES_DIR


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
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/test')

    sess.run(tf.local_variables_initializer())

    for epoch in range(num_epochs):
        _, loss_value = sess.run([optimizer, cross_entropy])
        if epoch % 10 == 0:
            train_summary = sess.run(merged)
            train_writer.add_summary(train_summary, epoch)
        test_summary = sess.run(merged, feed_dict={features: test_features, labels: test_labels})
        test_writer.add_summary(test_summary, epoch)


if __name__ == '__main__':
    delete_tensorboard_summaries()
    logistic_regression_training(num_epochs=5000, batch_size=50, training_step=0.2)
