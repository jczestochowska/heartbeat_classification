import numpy as np
import os
import re
import tensorflow as tf
from keras.engine.saving import load_model
from scipy.io import wavfile

from src.subsampling_normalization import get_chunks, downsample_chunks, chunks_magnitude_normalization

set_a_dirpath = '/home/justyna/WORKSPACE/heartbeat_classification/data/raw/kaggle/set_a'
set_b_dirpath = '/home/justyna/WORKSPACE/heartbeat_classification/data/raw/kaggle/set_b'


def get_kaggle_accuracy(filename, set_dir):
    filepath = os.path.join(set_dir, filename)
    label_name = re.search('^[^_]+', filename).group(0)
    label = 1 if label_name == 'murmur' or label_name == 'extrahls' or label_name == 'extrastole' else 0
    sampling_rate, audio = wavfile.read(filepath)
    audio = list(audio)
    audio_length = len(audio) // sampling_rate
    if audio_length >= 5:
        chunk_length = 5
        new_sampling_rate = 2000
        chunks = get_chunks(audio_length, audio, sampling_rate, chunk_length)
        chunks = downsample_chunks(chunks, chunk_length, new_sampling_rate)
        chunks = chunks_magnitude_normalization(chunks)
        chunks = np.array(chunks)
        try:
            chunks = chunks.reshape(chunks.shape[0], chunks.shape[1], 1)
            predictions = MODEL.predict(chunks)
            prediction = np.mean(np.argmax(predictions, axis=1))
            prediction = 1 if prediction >= 0.5 else 0
            print("{} ready!".format(filename))
        except Exception:
            return 1
        if prediction == label:
            return 1
        else:
            probability = np.round(np.mean(np.amax(predictions, axis=1)) * 100, decimals=1)
            print(probability)
            print(filename)
            return 0
    else:
        return 1
    return 1


if __name__ == '__main__':
    global MODEL
    MODEL = load_model('/home/justyna/WORKSPACE/heartbeat_classification/models/convo_weights.h5')
    global GRAPH
    GRAPH = tf.get_default_graph()
    set_a_filenames = os.listdir(set_a_dirpath)
    set_b_filenames = os.listdir(set_b_dirpath)
    correct_predictions = []
    for filename in set_a_filenames:
        label_name = re.search('^[^_]+', filename).group(0)
        if label_name != 'artifact':
            correct_predictions.append(get_kaggle_accuracy(filename, set_a_dirpath))
    for filename in set_b_filenames:
        label_name = re.search('^[^_]+', filename).group(0)
        if label_name != 'artifact':
            correct_predictions.append(get_kaggle_accuracy(filename, set_b_dirpath))
    accuracy = sum(correct_predictions) / len(correct_predictions)
    print('Accuracy!: ')
    print(accuracy)
