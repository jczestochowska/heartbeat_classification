import os
import tensorflow as tf
import threading
from flask import Flask, flash, request, render_template, url_for, after_this_request
from keras.engine.saving import load_model
from plotly import plotly
from werkzeug.utils import redirect

from prepare_report import preprocess_uploaded_file, get_prediction, save_plotly_report_to_html

PLOTLY_API_KEY = api_key = 'CzDrbDVsUbaHOC8eBV3d'
PLOTLY_USERNAME = username = 'j.czestochowska'

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSION = 'wav'

from config import UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Somethin is wrong I cannot find file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if not file.filename.endswith(ALLOWED_EXTENSION):
            flash('Please upload a .wav file')
            return redirect(request.url)
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


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    _load_model()
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
