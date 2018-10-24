import numpy as np
import os
import re
from keras.engine.saving import load_model
from scipy.io import wavfile

from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization

set_a_dirpath = '/home/justyna/WORKSPACE/heartbeat_classification/data/raw/kaggle/set_a'
set_b_dirpath = '/home/justyna/WORKSPACE/heartbeat_classification/data/raw/kaggle/set_b'
set_a_filenames = os.listdir(set_a_dirpath)
set_b_filenames = os.listdir(set_b_dirpath)

model = load_model('/home/justyna/WORKSPACE/heartbeat_classification/src/flask-app/convo_weights.h5')

filepath = os.path.join(set_a_dirpath, filename)
re.match('*_', filename)
sampling_rate, audio = wavfile.read(filepath)
audio_length = len(audio) // sampling_rate
chunk_length = 5
new_sampling_rate = 2000
chunks = get_chunks(audio_length, audio, sampling_rate, chunk_length)
chunks = downsample_chunks(chunks, chunk_length, new_sampling_rate)
chunks = chunks_magnitude_normalization(chunks)
chunks = np.array(chunks)
chunks = chunks.reshape(chunks.shape[0], chunks.shape[1], 1)
predictions = model.predict(chunks)
probability = np.round(np.mean(np.amax(predictions, axis=1)) * 100, decimals=1)
prediction = np.mean(np.argmax(predictions, axis=1))
prediction = 1 if prediction >= 0.5 else 0
