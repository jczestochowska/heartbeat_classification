import numpy as np
import os
import tensorflow as tf
import threading
from flask import Flask, request, render_template, url_for, after_this_request
from keras.engine.saving import load_model
from plotly import plotly
from werkzeug.utils import redirect

from prepare_report import preprocess_uploaded_file, get_prediction, save_plotly_report_to_html
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
    chunks, filepath, audio, sampling_rate = preprocess_uploaded_file()
    plotly_thread = threading.Thread(target=save_plotly_report_to_html, args=(audio, sampling_rate))
    plotly_thread.start()
    prediction, probability = get_prediction(chunks, MODEL, GRAPH)

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
    training_set = np.load('train.npy')
    explainer = LimeTimeSeriesExplanation(feature_selection='auto', verbose=False)
    explanations = explainer.explain_instance(timeseries=instance, num_features=num_features, training_set=training_set,
                                              num_samples=num_samples, num_slices=num_slices, classifier_fn=predict)
    return explanations


def predict(instances):
    labels = []
    for instance in instances:
        instance = np.reshape(instance, newshape=(1, instance.shape[0], 1))
        labels.append(MODEL.predict(instance))
    return np.array(labels).reshape(len(instances), 2)


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    _load_model()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
