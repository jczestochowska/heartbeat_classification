import numpy as np
import os
import plotly
import scipy
import tensorflow as tf
from keras.engine.saving import load_model
from plotly import graph_objs as go, tools as tls
from scipy import signal
from scipy.io import wavfile

from config import UPLOAD_FOLDER
from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization


def _load_model():
    global model
    model = load_model('convo_weights.h5')
    global graph
    return tf.get_default_graph(), model


def map_prediction_to_string(label):
    return "abnormal" if label == 1 else "normal"


def preprocess_uploaded_file():
    filepath = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[1])
    sampling_rate, audio = wavfile.read(filepath)
    audio = list(audio)
    audio_length = len(audio) // sampling_rate
    chunk_length = 5
    new_sampling_rate = 2000
    chunks = get_chunks(audio_length, audio, sampling_rate, chunk_length)
    chunks = downsample_chunks(chunks, chunk_length, new_sampling_rate)
    chunks = chunks_magnitude_normalization(chunks)
    chunks = np.array(chunks)
    return chunks.reshape(chunks.shape[0], chunks.shape[1], 1), filepath, audio, sampling_rate


def save_plotly_report_to_html(audio, sampling_rate):
    signal_plot_html_snippet, signal_plot_link = get_plotly_signal(audio)
    spectrogram_html_snippet, spectrogram_plot_link = get_plotly_spectrogram(audio, sampling_rate)
    with open('./templates/report.html', 'w') as file:
        file.write(signal_plot_html_snippet)
        file.write('\n')
        file.write(spectrogram_html_snippet)
        file.close()


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
    return tls.get_embed(plotly_link), plotly_link


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
    return tls.get_embed(plotly_link), plotly_link


def get_prediction(chunks):
    GRAPH, MODEL = _load_model()
    with GRAPH.as_default():
        predictions = MODEL.predict(chunks)
    probability = np.round(np.mean(np.amax(predictions, axis=1)) * 100, decimals=1)
    prediction = np.mean(np.argmax(predictions, axis=1))
    prediction = 1 if prediction >= 0.5 else 0
    prediction = map_prediction_to_string(prediction)
    return prediction, probability
