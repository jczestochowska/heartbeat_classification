import os

from flask import Flask, request, render_template

app = Flask(__name__)
ALLOWED_EXTENSIONS = ['wav']

UPLOAD_FOLDER = '/home/jczestochowska/flask-test'


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
