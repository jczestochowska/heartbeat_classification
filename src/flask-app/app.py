import os

import numpy as np
from flask import Flask, request, render_template
from plotly import plotly

from src.prepare_report import map_prediction_to_string, preprocess_uploaded_file, _load_model

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSIONS = ['wav']
GRAPH, MODEL = _load_model()

from config import UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        chunks = preprocess_uploaded_file()
        with GRAPH.as_default():
            predictions = MODEL.predict(chunks)
        prediction = np.mean(np.amax(predictions, axis=1))
        prediction = 1 if prediction >= 0.5 else 0
        prediction = map_prediction_to_string(prediction)
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(username='j.czestochowska', api_key='CzDrbDVsUbaHOC8eBV3d')
    app.run(debug=True)
