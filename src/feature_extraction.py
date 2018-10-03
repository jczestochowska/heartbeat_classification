from librosa import feature
from scipy.io import wavfile


def get_mfcc(filepath):
    sampling_rate, signal = wavfile.read(filepath)
    signal = signal.astype(float)
    return feature.mfcc(signal, sampling_rate)
