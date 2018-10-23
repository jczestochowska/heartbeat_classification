#!/bin/bash

apt-get update -y
apt-get install -y python-pip
pip install virtualenv
mkdir ~/.virtualenvs
pip install virtualenvwrapper

echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc

echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.profile
echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.profile

source ~/.bashrc
source ~/.profile

mkvirtualenv heartbeat -r requirements.txt --python=python3.5

