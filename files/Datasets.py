import librosa
import soundfile
from tqdm import tqdm
import h5py
import os
import torch
import numpy as np

from files.Utils import Utils
from torchvision import transforms
from files.FeatureExtractors import AudioFeatureExtractor
from torch.utils.data import DataLoader, Dataset


class SnippetProgDataset(Dataset):
    def load_file(file, snippet_length: int, sampling_rate: int) -> list:
        data, sr = librosa.load(file, sr = sampling_rate)
        data, trim_shape = librosa.effects.trim(data)
        sl = sr * snippet_length
        snippets = [data[i: i+sl] for i in range(0, len(data), sl)][:-1]
        return snippets
    
    
    def __init__(self, songs_array: list[tuple], snippet_length: int = 10, sampling_rate: int = 11025,  transform = None):
        self.transform = transform
        self.songs = songs_array
        self.data: list[tuple] = []
        self.labels: list[int] = []
        self.metadata: list[dict] = []
        for song_path, song_name, label in tqdm(self.songs):
            song = SnippetProgDataset.load_file(song_path, snippet_length, sampling_rate)
            for snippet_idx, snippet in enumerate(song):
                self.data.append(snippet)
                self.labels.append(label)
                self.metadata.append({'song_name': song_name, 'snippet_idx': snippet_idx})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        snippet = self.data[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            snippet = self.transform(snippet)

        return snippet, label, metadata
    
class WindowSnippetProgDataset(Dataset):
    def load_file(file, snippet_length: int, sampling_rate: int) -> list:
        data, sr = librosa.load(file, sr = sampling_rate)
        data, trim_shape = librosa.effects.trim(data)
        sl = sr * snippet_length
        snippets = [data[i: i+sl] for i in range(0, len(data), sl // 2)][:-2]
        return snippets
    
    
    def __init__(self, songs_array: list[tuple], snippet_length: int = 10, sampling_rate: int = 11025,  transform = None):
        self.transform = transform
        self.songs = songs_array
        self.data: list[tuple] = []
        self.labels: list[int] = []
        self.metadata: list[dict] = []
        for song_path, song_name, label in tqdm(self.songs):
            song = SnippetProgDataset.load_file(song_path, snippet_length, sampling_rate)
            for snippet_idx, snippet in enumerate(song):
                self.data.append(snippet)
                self.labels.append(label)
                self.metadata.append({'song_name': song_name, 'snippet_idx': snippet_idx})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        snippet = self.data[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            snippet = self.transform(snippet)

        return snippet, label, metadata
    
class RawAudioProgDataset(Dataset):
    def load_file(file, snippet_length:int, sampling_rate:int):
        data, sr = librosa.load(file, sr = sampling_rate)
        data, trim_shape = librosa.effects.trim(data)
        sl = sr * snippet_length
        snippets = [data[i: i+sl] for i in range(0, len(data), sl)][:-1]
        return snippets
    
    def __init__(self, songs_array: list[tuple], snippet_length: int = 10, sampling_rate: int = 11025, transform = None):
        self.snippet_length = snippet_length
        self.sampling_rate = sampling_rate
        self.transform = transform
        self.songs = songs_array
        self.data: list[tuple] = []
        self.metadata: list[dict] = []
        for song_path, song_name, label in tqdm(self.songs):
            song = RawAudioProgDataset.load_file(song_path, self.snippet_length, self.sampling_rate)
            for snippet_idx, snippet in enumerate(song):
                self.data.append((snippet, label))
                self.metadata.append({'song_name': song_name, 'snippet_idx': snippet_idx})

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        snippet, label = self.data[idx]
        metadata = self.metadata[idx]
        
        if self.transform:
            snippet, mfcc = self.transform(snippet)

        return snippet, mfcc, label, metadata
    