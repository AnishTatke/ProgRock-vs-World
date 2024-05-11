import torch
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import h5py
from pandas.plotting import table
from sklearn.metrics import confusion_matrix

class Utils:
    CLASS_NAMES = ['Non-Prog', 'Prog']
    getName = lambda s : s.split('\\')[-1]
    ifLocalFile = lambda s: s in os.listdir('datasets/')
    stringModel = lambda s : s.replace("\n", "<br/>").replace(" ", "&nsbp")
    getTime = lambda : (str(datetime.now()).split('.')[0]).replace(" ", "_")

    def save_to_h5py(data, labels, metadata, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('data', data=data, chunks=True, compression='gzip', compression_opts=9)
            f.create_dataset('labels', data=labels, chunks=True, compression='gzip', compression_opts=9)
            metadata_group = f.create_group('metadata')
            for i, entry in enumerate(metadata):
                entry_group = metadata_group.create_group(str(i))
                entry_group.create_dataset('song_name', data=entry['song_name'].encode('utf-8'))
                entry_group.create_dataset('snippet_idx', data=entry['snippet_idx'])
        f.close()
        print(f"Data saved to {filename}")

    def load_from_h5py(filename):
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
            labels = f['labels'][:]
            metadata = []
            for i, entry in f['metadata'].items():
                metadata.append({'song_name': entry['song_name'][()], 'snippet_idx': entry['snippet_idx'][()]})
        return data, labels, metadata
    
    def getConfusionMatrix(result: dict, predictions: list, labels: list) -> dict:
        labels = [i.item() for i in labels]
        if "conf_matrix" not in result:
            result['conf_matrix'] = np.zeros((2, 2))

        result['conf_matrix'] += confusion_matrix(labels, predictions)
        conf_matrix = result['conf_matrix']
        class_names = ['Non Prog', 'Prog']
        class_accuracy = {}
        for i in range(len(class_names)):
            class_accuracy[class_names[i]] = conf_matrix[i, i] / conf_matrix[i].sum() * 100

        for cls in class_names:
            result[f"{cls} Accuracy"] = class_accuracy[cls]        
        return result
    
    def splitSongs(split: float = 0.8, path: str = "", mySeed: int = 111):
        prog_folder = "Progressive_Rock_Songs" if path == "" else os.path.join(path + "//" + "Progressive_Rock_Songs")
        non_prog_folder = "Not_Progressive_Rock" if path == "" else os.path.join(path + "//" + "Not_Progressive_Rock")
        random.seed(mySeed)
        songs = [(i, Utils.getName(i), 1) for i in librosa.util.find_files(prog_folder)] + [(i, Utils.getName(i), 0) for i in librosa.util.find_files(non_prog_folder)]
        random.shuffle(songs)

        split_idx = int(split * len(songs))
        return songs[:split_idx], songs[split_idx:]
    
    def getHDF5Data(Dataset, songs: list, transform, set_name: str = "training", getNew: bool = False, saveData: bool = True):
        file_name = f"{set_name}_{Dataset.__name__}.h5"
        if getNew:
            dataset = Dataset(file_name, songs, [], [], [], False, transform = transform)
            if saveData:
                Utils.save_to_h5py(dataset.data, dataset.labels, dataset.metadata, f"datasets/{file_name}")
            return dataset
        else:
            if Utils.ifLocalFile(file_name):
                data, labels, metadata = Utils.load_from_h5py(f"datasets/{file_name}")
                dataset = Dataset(file_name, songs, data, labels, metadata, False, transform = transform)
                print(f"{file_name} loaded")
            else:
                print(f"{file_name} not found")
            return dataset
        
    def getPTData(Dataset, songs: list, transform, train: bool = True, getNew: bool = False, saveData: bool = True):
        file_name = f"training_{Dataset.__name__}.pt" if train else f"validation_{Dataset.__name__}.pt"
        if getNew:
            dataset = Dataset(songs, transform = transform)
            if saveData:
                try:        
                    torch.save(dataset, os.path.join(os.getcwd() + f"/datasets/{file_name}"))
                except:
                    print("Error saving data")
        else:
            dataset = torch.load(f"datasets/{file_name}") if Utils.ifLocalFile(file_name) else print(f"Not available")
        return dataset
    
    def getTestData(Dataset, test_songs: list, transform, getNew: bool = False, saveData: bool = False):
        if getNew:
            test_dataset = Dataset(test_songs, transform = transform)
        else:
            test_dataset = torch.load(f"datasets/test_{Dataset.__name__}.pt") if Utils.ifLocalFile(f"test_{Dataset.__name__}.pt") else print(f"Not available")

        if saveData:
            try:        
                torch.save(test_dataset, os.path.join(os.getcwd() + f"/datasets/test_{test_dataset.__class__.__name__}.pt"))
            except:
                print("Error saving data")
        return test_dataset
    
    def get_class_distribution(songs, set_name):
        for idx in range(len(Utils.CLASS_NAMES)):
            print(f"{set_name}|Number of {Utils.CLASS_NAMES[idx]} songs: {len([i for i in songs if i[2] == idx])}")
            class_counts = [len([i for i in songs if i[2] == idx]) for idx in range(len(Utils.CLASS_NAMES))]
        plt.pie(class_counts, labels=Utils.CLASS_NAMES, autopct='%1.1f%%')
        plt.title(f'{set_name} Set Class Distribution')
        plt.show()

    def updateSongDict(dictionary, metadata, outputs):
        for idx, song_name in enumerate(metadata['song_name']):
            if song_name not in dictionary:
                dictionary[song_name] = []
            dictionary[song_name].append((metadata['snippet_idx'][idx].item(), outputs[idx].item()))

    def saveModel(model, epoch, path = "models/"):
        file_name = f"model_{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, path + file_name)
        
    def plot_accuracies_and_losses(x_range: list, training: dict, validation: dict):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        if training and validation:
            if training["accuracies"] is not None and validation["accuracies"] is not None:
                axs[0].plot(x_range, training["accuracies"])
                axs[0].plot(x_range, validation["accuracies"], color="orange")
                axs[0].set_title('Accuracy')
                axs[0].set_xlabel('Epoch')
                axs[0].set_ylabel('Value')

            if training["losses"] is not None and validation["losses"] is not None:
                axs[1].plot(x_range, training["losses"])
                axs[1].plot(x_range, validation["losses"], color="orange")
                axs[1].set_title('Loss')
                axs[1].set_xlabel('Epoch')
                axs[1].set_ylabel('Value')

            plt.tight_layout()
            plt.show()
        else:
            print("No data to plot")

    def plot_confusion_matrix(conf_matrix, class_names):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(conf_matrix, cmap='Blues')

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("True Values")
        
        ax.set_title('Confusion Matrix')
        cbar = ax.figure.colorbar(im, ax=ax)
        plt.show()

    def getClassAccuracies(res, class_names):
        for key in res:
            for class_name in class_names:
                if class_name in key:
                    print(key, f"{res[key]:.2f}%")

    def plot_features(snippet, label, sr):
        mel_spec = snippet[:128, :]
        print(f"Mel-Spectogram: {mel_spec.shape}")
        mfcc = snippet[128 : 128 + 20, :]
        print(f"MFCC: {mfcc.shape}")
        chroma = snippet[128 + 20: 128 + 20 + 12, :]
        print(f"Chromagram: {chroma.shape}")

        # Spectrogram
        plt.figure(figsize=(10, 12))
        plt.suptitle(f"{Utils.CLASS_NAMES[label]} Features")
        
        plt.subplot(4, 1, 1)
        librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
        plt.title('Mel-Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        
        # Plot the MFCCs
        plt.subplot(4, 1, 2)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time')
        plt.title('MFCC')
        plt.colorbar()
        
        # Plot the chromagram
        plt.subplot(4, 1, 3)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.title('Chromagram')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()

    def getTable(df, title):
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        table(ax, df)
        plt.subplots_adjust(top=0.11, bottom=0.1)
        plt.title(title)
