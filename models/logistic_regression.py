import numpy as np
import os
import shutil
import tensorflow as tf

from config import PROJECT_ROOT_DIR

TRAIN = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'train.npy')
TEST = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized', 'test.npy')
TRAIN_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                            'train_labels.npy')
TEST_LABELS = os.path.join(PROJECT_ROOT_DIR, 'data', 'processed', 'preprocessed', 'physionet', 'serialized',
                           'test_labels.npy')
SUMMARIES_DIR = os.path.join(PROJECT_ROOT_DIR, 'models', 'tensorboard_summaries')


def main(num_epochs, batch_size=20, training_step=0.1):
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

    tf.summary.scalar('cross_entropy', cross_entropy)

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
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('precision', precision)
    tf.summary.scalar('recall', recall)

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

        print('-------------------------------------------------------------')
        print("Epoch: {}".format(epoch))
        print("training accuracy: ", sess.run(accuracy, feed_dict={features: train_features, labels: train_labels}))
        print("testing accuracy: ", sess.run(accuracy, feed_dict={features: test_features, labels: test_labels}))
        print("test precision: ", sess.run(precision_update, feed_dict={features: test_features, labels: test_labels}))
        print("test recall: ", sess.run(recall_update, feed_dict={features: test_features, labels: test_labels}))
        print('-------------------------------------------------------------')


def load_dataset():
    return np.load(TRAIN), np.load(TEST), np.load(TRAIN_LABELS), np.load(TEST_LABELS)


def get_balanced_dataset(train_features, train_labels):
    normal_data = np.squeeze(train_features[np.argwhere(train_labels == 0)], axis=1)
    abnormal_data = np.squeeze(train_features[np.argwhere(train_labels == 1)], axis=1)

    repeat = len(abnormal_data) - len(normal_data)
    normal_idx = np.argwhere(train_labels == 0).tolist()
    normal_data = train_features[np.concatenate((normal_idx, normal_idx[0:repeat]), axis=0)]
    normal_data = np.squeeze(normal_data, axis=1)

    normal_labels = np.zeros(shape=(normal_data.shape[0]))
    abnormal_labels = np.ones(shape=(abnormal_data.shape[0]))

    dataset_abnormal = tf.data.Dataset.from_tensor_slices((abnormal_data, abnormal_labels))
    dataset_normal = tf.data.Dataset.from_tensor_slices((normal_data, normal_labels))
    dataset = dataset_normal.concatenate(dataset_abnormal)
    return dataset.shuffle(buffer_size=train_features.shape[0])


def delete_tensorboard_summaries():
    train_summary = os.path.join(SUMMARIES_DIR, 'train')
    test_summary = os.path.join(SUMMARIES_DIR, 'test')
    if os.path.exists(train_summary) and os.path.exists(test_summary):
        shutil.rmtree(train_summary)
        shutil.rmtree(test_summary)


if __name__ == '__main__':
    delete_tensorboard_summaries()
    main(num_epochs=5000)
