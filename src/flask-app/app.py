import os

import numpy as np
from flask import Flask, request, render_template, url_for, after_this_request
from plotly import plotly
from werkzeug.utils import redirect

from src.prepare_report import map_prediction_to_string, preprocess_uploaded_file, _load_model

PLOTLY_API_KEY = api_key = 'CzDrbDVsUbaHOC8eBV3d'
PLOTLY_USERNAME = username = 'j.czestochowska'

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSIONS = ['wav']
GRAPH, MODEL = _load_model()

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
    chunks, filepath = preprocess_uploaded_file()

    @after_this_request
    def delete_file(response):
        os.remove(filepath)
        return response

    with GRAPH.as_default():
        predictions = MODEL.predict(chunks)
    probability = np.round(np.mean(np.amax(predictions, axis=1))*100, decimals=1)
    prediction = np.mean(np.argmax(predictions, axis=1))
    prediction = 1 if prediction >= 0.5 else 0
    prediction = map_prediction_to_string(prediction)
    return render_template('prediction.html', prediction=prediction, probability=probability)


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    app.run(debug=True)
