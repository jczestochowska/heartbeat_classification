# heartbeat_classification

simple web app that uses deep learning model to predict whether yout heartbeat is normal or abnormal from .wav records. This application was a subject of my bachelor thesis.

## prerequisities

* python 3.6
* preferably python virtual environment
* git

## getting started

### unix machine
to run app from command line on unix machine (if you use bash/zsh shell, it wouldn't work on e.g fish):

**PLEASE NOTICE**: 
repo stores almost ~1GB file containing training dataset, it was uploaded using git lfs, cloning may take some time

```
git clone https://github.com/jczestochowska/heartbeat_classification.git
*create virtual environment using requirements.txt file*
cd ./heartbeat_classification
export PYTHONPATH=$(pwd)
cd ./heartbeat_classification/src/flask-app
python3 app.py
```
now you have running instance on localhost,
heartbeat classifier is a flask app but please don't use flask cli as configuration is stored directly in app.py file

### using IDE
just open project in pycharm and run app.py file


