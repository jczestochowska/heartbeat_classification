import time

import numpy as np
import os
import tensorflow as tf
from flask import Flask, request, render_template, url_for, after_this_request
from keras.engine.saving import load_model
from multiprocessing.pool import ThreadPool
from plotly import plotly
from werkzeug.utils import redirect

from prepare_report import preprocess_uploaded_file, get_prediction, plot_lime_explanation, \
    get_plotly_signal, get_plotly_spectrogram
from src.lime_timeseries_optimized import LimeTimeSeriesExplanation

PLOTLY_API_KEY = api_key = 'CzDrbDVsUbaHOC8eBV3d'
PLOTLY_USERNAME = username = 'j.czestochowska'

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSION = 'wav'

from config import UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        return redirect(url_for('predict'))
    return render_template("index.html")


@app.route('/predict', methods=['GET'])
def predict():
    start = time.time()
    chunks, filepath, audio, sampling_rate = preprocess_uploaded_file()
    chunks_for_lime = np.squeeze(chunks)

    signal_thread = ThreadPool(processes=1)
    spectrogram_thread = ThreadPool(processes=1)
    lime_thread = ThreadPool(processes=10)

    lime_pool_result = lime_thread.map_async(get_lime_explanation, chunks_for_lime)
    signal_pool_result = signal_thread.apply_async(get_plotly_signal, args=([audio]))
    spectrogram_pool_result = spectrogram_thread.apply_async(get_plotly_spectrogram, args=(audio, sampling_rate))

    prediction, probability = get_prediction(chunks, MODEL, GRAPH)

    lime_htmls = lime_pool_result.get()
    signal_html = signal_pool_result.get()
    spectrogram_html = spectrogram_pool_result.get()
    save_htmls_to_file(lime_htmls, signal_html, spectrogram_html)
    end = time.time()
    print(end - start)

    @after_this_request
    def delete_file(response):
        os.remove(filepath)
        return response

    return render_template('prediction.html', prediction=prediction, probability=probability)


def _load_model():
    global MODEL
    MODEL = load_model('convo_weights.h5')
    global GRAPH
    GRAPH = tf.get_default_graph()


def get_lime_explanation(instance, num_features=20, num_slices=30, num_samples=100):
    explainer = LimeTimeSeriesExplanation(feature_selection='auto', verbose=False)
    explanations = explainer.explain_instance(timeseries=instance, num_features=num_features, training_set=TRAINING_SET,
                                              num_samples=num_samples, num_slices=num_slices,
                                              classifier_fn=lime_predict)
    return plot_lime_explanation(explanations, instance)


def lime_predict(instances):
    labels = []
    for instance in instances:
        instance = np.reshape(instance, newshape=(1, instance.shape[0], 1))
        with GRAPH.as_default():
            label = MODEL.predict(instance)
        labels.append(label)
    return np.array(labels).reshape(len(instances), 2)


def save_htmls_to_file(lime_htmls, signal_html, spectrogram_html):
    with open('./templates/report.html', 'w') as file:
        for html in lime_htmls:
            file.write(html)
            file.write('\n')
        file.write(signal_html)
        file.write('\n')
        file.write(spectrogram_html)


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    _load_model()
    global TRAINING_SET
    TRAINING_SET = np.load('train.npy')
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, port=5002)
