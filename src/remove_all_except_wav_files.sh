#!/bin/bash
DIRECTORY=/home/justyna/WORKSPACE/heartbeat_classification/data/raw/physionet
echo $DIRECTORY
find $DIRECTORY -type f  ! -name '*.wav' -delete