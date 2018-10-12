import os

from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='/static')
ALLOWED_EXTENSIONS = ['wav']

from config import UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
