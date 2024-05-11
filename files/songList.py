import os
import ast
import librosa
import pandas as pd
from files.Utils import Utils
CLASS_NAMES = ["Non-Prog", "Prog"]
FOLDERS = ["Not_Progressive_Rock", "Progressive_Rock_Songs"]
getString = lambda s: ast.literal_eval(s).decode('utf-8')

def generate_song_csv(name: str, songs: list):
    ssongs = [(path.encode('utf-8'), name.encode('utf-8'), label) for path, name, label in songs]
    df = pd.DataFrame(ssongs, columns = ["Path", "File Name", "Label"])
    df.to_csv(f"{name}_audio_files.csv", index = False)

def generate_table(file_name: str, songs: list):
    df = pd.DataFrame(songs, columns = ["Name", "True Label", "Prediction", "Prediction Score"])
    df.to_csv(f"tables/{file_name}.csv", index = False)

def load_csv(file_path):
    df = pd.read_csv(file_path)
    paths = [getString(path) for path in df["Path"].tolist()]
    file_names = [getString(name) for name in df["File Name"].tolist()]
    labels = df["Label"].tolist()
    return list(zip(paths, file_names, labels))
