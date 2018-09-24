import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

from src.dataset_getters import MERGED_DATASETS_PATH

data_dir_path = MERGED_DATASETS_PATH


def get_one_second_chunks(sampling_rate, signal):
    audio_length = len(signal) // sampling_rate
    return [signal[i * sampling_rate:(i + 1) * sampling_rate] for i in range(audio_length)]


def downsample_chunks(chunks, new_sampling_rate=2000):
    return [resample(chunk, new_sampling_rate) for chunk in chunks]


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
