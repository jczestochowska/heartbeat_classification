import time

import numpy as np
import os
import tensorflow as tf
import uuid
from flask import Flask, request, render_template, url_for, after_this_request, flash, session
from keras.engine.saving import load_model
from multiprocessing.pool import ThreadPool
from plotly import plotly
from scipy.io import wavfile
from werkzeug.utils import redirect

from prepare_report import preprocess_uploaded_file, get_prediction, get_plotly_signal, get_plotly_spectrogram, \
    plot_lime_explanation
from src.lime_timeseries_optimized import LimeTimeSeriesExplanation

PLOTLY_API_KEY = api_key = 'CzDrbDVsUbaHOC8eBV3d'
PLOTLY_USERNAME = username = 'j.czestochowska'

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSION = 'wav'

from config import UPLOAD_FOLDER


@app.errorhandler(404)
def page_not_found(e):
    return render_template('not_found.html')


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('internal_server_error.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template("index.html")
        file = request.files['file']
        if not file.filename.endswith('.wav'):
            flash("Please upload a .wav file")
            return render_template("index.html")
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        session['filepath'] = filepath
        session['plots_uuid'] = uuid.uuid4().hex
        global AUDIO
        global SAMPLING_RATE
        SAMPLING_RATE, AUDIO = wavfile.read(filepath)
        AUDIO = list(AUDIO)
        global AUDIO_LENGTH
        AUDIO_LENGTH = len(AUDIO) // SAMPLING_RATE
        if AUDIO_LENGTH < 5:
            flash("Please upload audio that lasts at least 5 seconds")
            return render_template("index.html")
        return redirect(url_for('predict'))
    return render_template("index.html")


@app.route('/predict', methods=['GET'])
def predict():
    start = time.time()
    filepath = session.get('filepath')
    plots_uuid = session.get('plots_uuid')
    chunks = preprocess_uploaded_file(AUDIO, SAMPLING_RATE, AUDIO_LENGTH)
    chunks_for_lime = chunks
    if chunks.shape[0] > 1:
        chunks_for_lime = np.squeeze(chunks)
    elif chunks.shape[0] == 1:
        chunks_for_lime = np.squeeze(chunks, axis=2)
    signal_thread = ThreadPool(processes=1)
    spectrogram_thread = ThreadPool(processes=1)
    lime_thread = ThreadPool(processes=25)

    lime_pool_result = lime_thread.map_async(get_lime_explanation, chunks_for_lime)
    signal_thread.apply_async(get_plotly_signal, args=(AUDIO, plots_uuid))
    spectrogram_thread.apply_async(get_plotly_spectrogram, args=(AUDIO, SAMPLING_RATE, plots_uuid))

    prediction, probability = get_prediction(chunks, MODEL, GRAPH)
    lime_htmls = lime_pool_result.get()
    save_htmls_to_file(lime_htmls, plots_uuid)
    end = time.time()
    print(end - start)

    @after_this_request
    def delete_file(response):
        os.remove(filepath)
        os.remove('./templates/signal' + plots_uuid + '.html')
        os.remove('./templates/spectrogram' + plots_uuid + '.html')
        os.remove('./templates/lime' + plots_uuid + '.html')
        return response

    return render_template('prediction.html', prediction=prediction, probability=probability, plots_uuid=plots_uuid)


def _load_model():
    global MODEL
    MODEL = load_model('convo_weights.h5')
    global GRAPH
    GRAPH = tf.get_default_graph()


def get_lime_explanation(instance, num_features=10, num_slices=40, num_samples=250):
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


def save_htmls_to_file(lime_htmls, plots_uuid):
    chunks_number = len(lime_htmls)
    with open('./templates/lime' + plots_uuid + '.html', 'w') as file:
        for index, html in enumerate(lime_htmls, start=1):
            file.write('{}/{}'.format(index, chunks_number))
            file.write(html)
            file.write('\n')


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    _load_model()
    global TRAINING_SET
    TRAINING_SET = np.load('train.npy')
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True, port=5002)
