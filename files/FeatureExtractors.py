import librosa
import torch
import numpy as np
import soundfile
import audioread

class AudioFeatureExtractor():
    def __init__(self, sr: int):
        self.sr = sr
    def __call__(self, snippet):
        snippet = librosa.util.normalize(snippet)

        mel_spec = librosa.feature.melspectrogram(y = snippet, n_mels = 128)
        mfcc = librosa.feature.mfcc(y = snippet)
        chroma = librosa.feature.chroma_stft(y = snippet)
        features = np.concatenate((mel_spec, mfcc, chroma), axis=0)
        return features
    
class SpecAudioFeatureExtractor():
    def __init__(self, sr: int):
        self.sr = sr

    def __call__(self, snippet):
        snippet = librosa.util.normalize(snippet)

        mel_spec = librosa.feature.melspectrogram(y = snippet, n_mels = 128)
        mfcc = librosa.feature.mfcc(y = snippet)
        chroma = librosa.feature.chroma_stft(y = snippet)
        spec_cent = librosa.feature.spectral_centroid(y=snippet, sr=self.sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=snippet, sr=self.sr)
        tempo = librosa.feature.tempogram(y=snippet, sr=self.sr)
        onset_strength = librosa.onset.onset_strength(y=snippet, sr=self.sr).reshape(1, -1)
        rolloff = librosa.feature.spectral_rolloff(y=snippet, sr=self.sr)
        features = np.concatenate((mel_spec, mfcc, chroma, spec_cent, spec_bw, tempo, onset_strength, rolloff), axis=0)   
        return features
    
class RawDataAudioFeatureExtractor():
    def __init__(self):
        pass

    def __call__(self, snippet):
        #Librosa Transforms
        snippet = librosa.util.normalize(snippet)
        mfcc = librosa.feature.mfcc(y = snippet)

        # Tensor Transforms
        snippet = torch.tensor(snippet).view(1, -1)
        return snippet, mfcc