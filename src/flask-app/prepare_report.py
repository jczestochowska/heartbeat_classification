import math

import numpy as np
import plotly
import scipy
from plotly import graph_objs as go
from plotly import tools as tls
from scipy import signal

from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization


def map_prediction_to_string(label):
    return "abnormal" if label == 1 else "normal"


def preprocess_uploaded_file(audio, sampling_rate, audio_length):
    chunk_length = 5
    new_sampling_rate = 2000
    chunks = get_chunks(audio_length, audio, sampling_rate, chunk_length)
    chunks = downsample_chunks(chunks, chunk_length, new_sampling_rate)
    chunks = chunks_magnitude_normalization(chunks)
    chunks = np.array(chunks)
    return chunks.reshape(chunks.shape[0], chunks.shape[1], 1)


def get_plotly_signal(audio):
    audio = scipy.signal.decimate(audio, 6)
    x = np.linspace(0, len(audio), len(audio))
    layout = go.Layout(
        title='Your heartbeat signal',
        yaxis=dict(title='Magnitude'),  # x-axis label
        xaxis=dict(title='Sample'),  # y-axis label
    )
    data = [go.Scattergl(x=x, y=audio)]
    fig = go.Figure(data=data, layout=layout)
    plotly_link = plotly.plotly.plot(fig, auto_open=False)
    return tls.get_embed(plotly_link)


def get_plotly_spectrogram(audio, sampling_rate):
    audio = scipy.signal.decimate(audio, 2)
    audio = np.array(audio)
    freqs, bins, Pxx = signal.spectrogram(audio, fs=sampling_rate)

    trace = [go.Heatmap(
        x=bins,
        y=freqs,
        z=10 * np.log10(Pxx),
        colorscale='Viridis',
    )]
    layout = go.Layout(
        title='Spectrogram',
        yaxis=dict(title='Frequency'),
        xaxis=dict(title='Time'),
    )
    fig = go.Figure(data=trace, layout=layout)
    plotly_link = plotly.plotly.plot(fig, filename='Spectrogram', auto_open=False)
    return tls.get_embed(plotly_link)


def plot_lime_explanation(explanations, instance, num_slices=40):
    exp = explanations[0][1]
    trace = go.Scatter(
        x=np.arange(0, 10000, 1),
        y=instance,
        mode='lines',
    )
    data = [trace]
    layout = {'title': 'Samples that influenced classifiers decision', 'xaxis': {'title': 'Sample', 'showgrid': False},
              'yaxis': {'title': 'Magnitude', 'showgrid': False}, 'shapes': []}
    shape = {'type': 'rect', 'xref': 'x', 'yref': 'paper', 'x0': 0, 'y0': 0, 'x1': 0, 'y1': 1, 'fillcolor': '#f24d50',
             'opacity': 0.0, 'line': {'width': 0}, 'layer': 'below'}
    values_per_slice = math.ceil(len(instance) / num_slices)
    weights = [abs(sample[1]) for sample in exp]
    normalized_weights = [(weight - min(weights)) / (max(weights) - min(weights)) for weight in weights]
    for i in range(len(exp)):
        feature, _ = exp[i]
        weight = normalized_weights[i]
        start = feature * values_per_slice
        end = start + values_per_slice
        shape1 = shape.copy()
        shape1.update({'x0': start, 'x1': end, 'opacity': weight})
        layout['shapes'].append(shape1)
    fig = go.Figure(data=data, layout=layout)
    lime_plot_link = plotly.plotly.plot(fig, auto_open=False)
    return tls.get_embed(lime_plot_link)


def get_prediction(chunks, model, graph):
    with graph.as_default():
        predictions = model.predict(chunks)
    probability = np.round(np.mean(np.amax(predictions, axis=1)) * 100, decimals=1)
    prediction = np.mean(np.argmax(predictions, axis=1))
    prediction = 1 if prediction >= 0.5 else 0
    prediction = map_prediction_to_string(prediction)
    return prediction, probability
