import asyncio
import os
from flask import Flask, flash, request, render_template, url_for, after_this_request
from plotly import plotly
from werkzeug.utils import redirect

from src.prepare_report import preprocess_uploaded_file, save_plotly_report_to_html, get_prediction

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

    loop = asyncio.get_event_loop()
    tasks = [
        get_prediction(chunks),
        save_plotly_report_to_html(audio, sampling_rate)
    ]
    prediction, probability = loop.run_until_complete(asyncio.gather(*tasks))[0]

    # save_plotly_report_to_html(audio, sampling_rate)
    # prediction, probability = get_prediction(chunks)
    @after_this_request
    def delete_file(response):
        os.remove(filepath)
        return response
    return render_template('prediction.html', prediction=prediction, probability=probability)


if __name__ == '__main__':
    plotly.plotly.tools.set_credentials_file(PLOTLY_USERNAME, PLOTLY_API_KEY)
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
