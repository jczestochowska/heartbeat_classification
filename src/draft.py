import math
from multiprocessing.pool import Pool

import numpy as np
import tensorflow as tf
from librosa import feature
from plotly.offline import plot
import plotly.graph_objs as go
from tensorflow.python.saved_model import tag_constants

from src.lime_timeseries_optimized import LimeTimeSeriesExplanation


def plot_lime_explanation(explanations, instance, prediction, label, error_idx, num_slices=40):
    exp = explanations.as_list(label=0)

    trace = go.Scatter(
        x=np.arange(0, 10000, 1),
        y=np.squeeze(instance),
        mode="lines"
    )
    data = [trace]
    if prediction == 1 and label == 0:
        mistake = 'FN'
    elif prediction == 0 and label == 1:
        mistake = 'FP'
    layout = {'font': {'size': 20},
              'title': 'Predykcja: {}, Etykieta: {}, Rodzaj błędu: {}'.format(prediction, label, mistake),
              'xaxis': {'title': 'Próbka', 'showgrid': False},
              'yaxis': {'title': 'Natężenie dźwięku', 'showgrid': False}, 'shapes': []}
    shape = {'type': 'rect', 'xref': 'x', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 1, 'fillcolor': '#f24d50',
             'opacity': 0.0, 'line': {'width': 0}, 'layer': 'below'}
    values_per_slice = math.ceil(len(instance) / num_slices)
    weights = [abs(sample[1]) * 10 ** 25 for sample in exp]
    normalized_weights = [(weight - min(weights)) / (max(weights) - min(weights)) for weight in weights]
    for i in range(len(exp)):
        feature, _ = exp[i]
        weight = normalized_weights[i]
        if weight < 0.1:
            weight = 0.1
        start = feature * values_per_slice
        end = start + values_per_slice
        shape1 = shape.copy()
        shape1.update({'x0': start, 'x1': end, 'opacity': weight})
        layout['shapes'].append(shape1)
    fig = go.Figure(data=data, layout=layout)
    #     pio.write_image(fig, 'images/error_{}.png'.format(error_idx))
    return plot(fig, filename='exp_lr.html', output_type='div')


def lime_lr_predict(instances):
    predictions = []
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(
                sess,
                [tag_constants.SERVING],
                "/home/jczestochowska/workspace/heartbeat_classification/models/logistic_regression_weights"
            )
            for instance in instances:
                instance = feature.mfcc(instance, 2000)
                w = graph.get_tensor_by_name("Variable:0")
                b = graph.get_tensor_by_name("Variable_1:0")
                instance = instance.reshape(1, 400)
                logits = tf.matmul(instance, w) + b
                prediction = sess.run(tf.sigmoid(logits))
                predictions.append(np.argmax(prediction))
    return predictions


if __name__ == '__main__':
    test_set = np.load(
        '/home/jczestochowska/workspace/heartbeat_classification/data/processed/preprocessed/physionet/serialized/no_feature_extraction/test.npy')
    test_labels = np.load(
        '/home/jczestochowska/workspace/heartbeat_classification/data/processed/preprocessed/physionet/serialized/no_feature_extraction/test_labels.npy')
    TRAINING_SET = np.load(
        '/home/jczestochowska/workspace/heartbeat_classification/data/processed/preprocessed/physionet/serialized/no_feature_extraction/train.npy')
    predictions_lr = lime_lr_predict(test_set[0:3])
    errors_lr = np.where(predictions_lr != test_labels)
    errors_lr = errors_lr.reshape(test_set.shape[0], 2)
    error_indices = np.save("logistic_regression_error_predictions_indices.npy", errors_lr)
# error_idx = 1
explainer = LimeTimeSeriesExplanation(feature_selection='auto', verbose=False)
# explanations = explainer.explain_instance(timeseries=test_set[error_idx], num_features=10,
#                                           training_set=TRAINING_SET[:100],
#                                           num_samples=250, num_slices=40,
#                                           classifier_fn=lime_lr_predict)
# plot_lime_explanation(explanations=explanations, error_idx=error_idx, prediction=predictions_lr[error_idx],
#                       label=test_labels[error_idx], instance=test_set[error_idx])
