import numpy as np
import os
from flask import Flask, request, render_template
from keras import backend
from keras.engine.saving import load_model

from src.prepare_report import map_prediction_to_string, preprocess_uploaded_file

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSIONS = ['wav']

from config import UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    chunks = preprocess_uploaded_file()
    model = load_model('convo_weights.h5')
    predictions = model.predict(chunks)
    prediction = np.mean(np.amax(predictions, axis=1))
    if prediction == 0.5:
        prediction = 1
    else:
        prediction = int(round(prediction))
    prediction = map_prediction_to_string(prediction)
    backend.clear_session()
    return render_template('prediction.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
