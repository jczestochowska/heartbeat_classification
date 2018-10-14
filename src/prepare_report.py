import numpy as np
import os
import plotly
from plotly import graph_objs as go, tools as tls
from scipy import signal
from scipy.io import wavfile

from config import UPLOAD_FOLDER
from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization


def map_prediction_to_string(label):
    return "abnormal" if label == 1 else "normal"


def preprocess_uploaded_file():
    filepath = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])
    sampling_rate, audio = wavfile.read(filepath)
    audio = list(audio)
    audio_length = len(audio) // sampling_rate
    chunk_length = 5
    new_sampling_rate = 2000
    chunks = get_chunks(audio_length, audio, sampling_rate, chunk_length)
    chunks = downsample_chunks(chunks, chunk_length, new_sampling_rate)
    chunks = chunks_magnitude_normalization(chunks)
    chunks = np.array(chunks)
    save_plotly_report_to_html(audio, sampling_rate)
    return chunks.reshape(chunks.shape[0], chunks.shape[1], 1)


def save_plotly_report_to_html(audio, sampling_rate):
    html_snippet = get_plotly_signal(audio)
    html_snippet1 = get_plotly_spectrogram(audio, sampling_rate)
    with open('./templates/report.html', 'w') as file:
        file.write(html_snippet)
        file.write('\n')
        file.write(html_snippet1)
        file.close()


def get_plotly_signal(audio):
    x = np.linspace(0, 1, len(audio))
    layout = go.Layout(
        title='Your heartbeat signal',
        yaxis=dict(title='Magnitude'),  # x-axis label
        xaxis=dict(title='Sample'),  # y-axis label
    )
    data = [go.Scatter(x=x, y=audio)]
    fig = go.Figure(data=data, layout=layout)
    plotly_link = plotly.plotly.plot(fig, auto_open=False)
    return tls.get_embed(plotly_link)


def get_plotly_spectrogram(audio, sampling_rate):
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
