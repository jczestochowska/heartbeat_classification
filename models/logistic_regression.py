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


def main(num_epochs, batch_size=10, training_step=0.1):
    train_features, test_features, train_labels, test_labels = load_dataset()

    balanced_dataset = get_balanced_dataset(train_features, train_labels)
    balanced_dataset = balanced_dataset.repeat().batch(batch_size)

    sess = tf.InteractiveSession()
    iterator = balanced_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    labels = tf.cast(labels, tf.int64)

    weights = tf.Variable(tf.random_normal([train_features.shape[1], 2], dtype=tf.float64))
    bias = tf.Variable(tf.random_normal([2], dtype=tf.float64))

    logits = tf.matmul(features, weights) + bias

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(training_step).minimize(cross_entropy)

    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        _, loss_value = sess.run([optimizer, cross_entropy])
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('-------------------------------------------------------------')
        print("Epoch: {}".format(epoch))
        print("training accuracy: ", sess.run(accuracy, feed_dict={features: train_features, labels: train_labels}))
        print("testing accuracy: ", sess.run(accuracy, feed_dict={features: test_features, labels: test_labels}))
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


if __name__ == '__main__':
    main(num_epochs=1000)
