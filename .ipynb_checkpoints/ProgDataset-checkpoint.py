import os
import numpy as np
import torch
import librosa
import soundfile
from tqdm import tqdm

from torch.utils.data import Dataset

class SnippetProgDataset(Dataset):
    def load_file(file, snippet_length = 10, sampling_rate = 11025):
        data, sr = librosa.load(file, sr = sampling_rate)
        data, trim_shape = librosa.effects.trim(data)
        sl = sr * snippet_length
        snippets = [data[i: i+sl] for i in range(0, len(data), sl)]
        feature_snippets = []
        for idx, snippet in enumerate(snippets):
            if idx == len(snippets) - 1:
                break
            feature_snippets.append(SnippetProgDataset.extract_features(snippet))
        return feature_snippets

    def extract_features(snippet):
        magnitude = np.abs(librosa.stft(snippet))
        mel_spec = librosa.feature.melspectrogram(S = magnitude**2)
        mfcc = librosa.feature.mfcc(S = librosa.power_to_db(mel_spec))
        chroma = librosa.feature.chroma_stft(S = magnitude)
        # tempo, beat_frames = librosa.beat.beat_track(y = snippet, sr = sr)
        # cq_chroma = librosa.feature.chroma_cqt(y = snippet, sr = sr)
        # onset_env = librosa.onset.onset_strength(y = snippet, sr=sr)
        features = np.concatenate((mel_spec, mfcc, chroma), axis=0)
        return features
    
    def __init__(self, songs_array, transform = None):
        self.transform = transform
        self.songs = songs_array
        self.data = []
        self.metadata = []
        for song_path, song_name, label in tqdm(self.songs):
            song = SnippetProgDataset.load_file(song_path)
            for snippet_idx, snippet in enumerate(song):
                self.data.append((snippet, label))
                self.metadata.append({'song_name': song_name, 'snippet_idx': snippet_idx})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        snippet, label = self.data[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            snippet = self.transform(snippet)

        return snippet, label, metadata