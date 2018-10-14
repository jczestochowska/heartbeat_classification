import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

from src.dataset_getters import MERGED_DATASETS_PATH

data_dir_path = MERGED_DATASETS_PATH


def get_chunks(audio_length, signal, sampling_rate, chunk_length=5):
    chunks_number = round(audio_length / chunk_length)
    if (audio_length / chunk_length) > 1:
        samples_missing = (chunk_length * sampling_rate - len(signal[chunk_length * sampling_rate:]))
        signal += signal[0:samples_missing]
    return [signal[i * chunk_length * sampling_rate:(i + 1) * chunk_length * sampling_rate] for i in
            range(chunks_number)]


def downsample_chunks(chunks, chunk_length=5, new_sampling_rate=4000):
    return [resample(chunk, chunk_length * new_sampling_rate) for chunk in chunks]


def chunks_magnitude_normalization(chunks):
    return [standard_score(chunk) for chunk in chunks]


def standard_score(chunk):
    series = pd.Series(chunk)
    values = series.values
    values = values.reshape((len(values), 1))
    scaler = StandardScaler()
    scaler = scaler.fit(values)
    normalized = scaler.transform(values)
    return normalized.reshape(len(values)).tolist()
