import torch
import torch.nn as nn
import torch.nn.functional as F

from files.Utils import Utils

def prepareSongData(dictionary, songs, model='rnn'):
    getSongLabel = lambda name : [song[2] for song in songs if song[1] == name][0]
    shape = (-1, 1, 1) if model == "rnn" else (-1, 1)
    sortingKey = lambda x: x[0]
    snippets_list = []
    label_list = []
    with torch.no_grad():
        for key in dictionary:
            snippets_list.append(torch.tensor([output[1] for output in sorted(dictionary[key], key=sortingKey)]).unsqueeze_(0).view(shape))
            label_list.append(torch.tensor(getSongLabel(key)).view(-1, 1))        
    return snippets_list, label_list

# Simple RNN
class PredictionsRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictionsRNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i20 = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        return F.sigmoid(self.i20(combined)), self.i2h(combined)
    
    def init_hidden(self, device):
        return torch.zeros(1, self.hidden_size).to(device)

def train_one_song(model, criterion, optimizer, song, label, device):
    hidden = model.init_hidden(device)
    for i in range(song.shape[0]):
        output, hidden = model(song[i], hidden)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()

def validate_one_song(model, criterion, song, label, device):
    hidden = model.init_hidden(device)
    with torch.no_grad():
        for i in range(song.shape[0]):
            output, hidden = model(song[i], hidden)
        loss = criterion(output, label)
    return output, loss.item()



def train_rnn(epoch: int, songs: list, labels: list, model, criterion, optimizer, device) -> tuple:
    model.train()
    running_loss = 0.0
    correct = 0
    for song, label in zip(songs, labels):
        song, label = song.float().to(device), label.float().to(device)
        output, loss = train_one_song(model, criterion, optimizer, song, label, device)
        correct += ((output > 0.5).long().item() == label.item()).real
        running_loss += loss

    accuracy = correct / len(songs) * 100
    running_loss /= len(songs)
    print(f"Epoch {epoch}: Epochwise Loss: {running_loss:.4f}\tEpochwise Accuracy: {correct}/{len(songs)} {accuracy:.2f}%")
    return accuracy, running_loss

def validate_rnn(epoch: int, songs: list, labels: list, model, criterion, device) -> tuple:
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for song, label in zip(songs, labels):
            song, label = song.float().to(device), label.float().to(device)
            output, loss = validate_one_song(model, criterion, song, label, device)
            correct += ((output > 0.5).long().item() == label.item()).real
            running_loss += loss
        
        accuracy = correct / len(songs) * 100
        running_loss /= len(songs)
        print(f"Epoch {epoch}: Validation Loss: {running_loss:.4f}\tValidation Accuracy: {correct}/{len(songs)} {accuracy:.2f}%")
    return accuracy, running_loss

def evaluate_rnn(song_names: list, songs: list, labels: list, model, criterion, device) -> tuple:
    model.eval()
    running_loss = 0.0
    correct = 0
    predictions = []
    mis_classified_count = 0
    songs_table = []
    with torch.no_grad():
        for idx, (song, label) in enumerate(zip(songs, labels)):
            song, label = song.float().to(device), label.float().to(device)
            output, loss = validate_one_song(model, criterion, song, label, device)
            prediction = (output > 0.5).long().item()
            if prediction != label.item():
                mis_classified_count += 1
                print(f"{mis_classified_count})\t{song_names[idx]}\t {prediction}:{int(label.item())}\t{round(output.item(), 2)}")
                songs_table.append([song_names[idx], int(label.item()), prediction, round(output.item(), 2)])
            predictions.append(prediction)
            correct += (prediction == label.item()).real
            running_loss += loss
        
        dictn = Utils.getConfusionMatrix({}, predictions, labels)
        accuracy = correct / len(songs) * 100
        running_loss /= len(songs)
        print(f"Evaluation: Loss: {running_loss:.4f}\tAccuracy: {correct}/{len(songs)} {accuracy:.2f}%")

    return accuracy, running_loss, dictn, songs_table

def eval_score_rnn(song_names: list, songs: list, model, device):
    model.eval()
    with torch.no_grad():
        for idx, song, in enumerate(songs):
            song = song.float().to(device)
            hidden = model.init_hidden(device)
            for i in range(song.shape[0]):
                output, hidden = model(song[i], hidden)
            print(f"{idx}\t{song_names[idx]}\t{output.item():.2f}")

# LSTM Network
class MyLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, num_classes: int = 1):
        super(MyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        out, _ = self.lstm(x.unsqueeze(0), (h0.unsqueeze(0), c0.unsqueeze(0)))
        out = self.fc(out[:, -1, :])
        return F.sigmoid(out)
        
def train_lstm(epoch: int, songs: list, labels: list, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    for song, label in zip(songs, labels):
        song, label = song.float().to(device), label.float().to(device)
        output = model(song)
        loss = criterion(output, label)
        correct += ((output > 0.5).long() == label).real
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = correct.item()
    accuracy = (100. * (correct / len(songs)))
    running_loss /= len(songs)
    print(f"Epoch {epoch}: Epochwise Loss: {running_loss:.4f}\tEpochwise Accuracy: {correct}/{len(songs)} {accuracy:.2f}%")
    return accuracy, running_loss

def validate_lstm(epoch: int, songs: list, labels: list, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for song, label in zip(songs, labels):
            song, label = song.float().to(device), label.float().to(device)
            output = model(song)
            loss = criterion(output, label)

            correct += ((output > 0.5).long().item() == label.item()).real
            running_loss += loss.item()

        accuracy = 100. * (correct / len(songs))
        running_loss /= len(songs)
        print(f"Epoch {epoch}: Validation Loss: {running_loss:.4f}\tValidation Accuracy: {correct}/{len(songs)} {accuracy:.2f}%")
    return accuracy, running_loss

def evaluate_lstm(song_names: list, songs: list, labels: list, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    predictions = []
    dictn = {}
    mis_classified_count = 0
    songs_table = []
    with torch.no_grad():
        for idx, (song, label) in enumerate(zip(songs, labels)):
            song, label = song.float().to(device), label.float().to(device)
            output = model(song)
            loss = criterion(output, label)

            prediction = (output > 0.5).long().item()
            if prediction != label.item():
                mis_classified_count += 1
                print(f"{mis_classified_count})\t{song_names[idx]}\t {prediction}:{int(label.item())}\t{round(output.item(), 2)}")
                songs_table.append([song_names[idx], int(label.item()), prediction, round(output.item(), 2)])
            predictions.append(prediction)
            correct += (prediction == label.item()).real
            running_loss += loss.item()

        dictn = Utils.getConfusionMatrix(dictn, predictions, labels)
        accuracy = 100. * (correct / len(songs))
        running_loss /= len(songs)
        print(f"Evaluation: Loss: {running_loss:.4f}\tAccuracy: {correct}/{len(songs)} {accuracy:.2f}%")
    return accuracy, running_loss, dictn, songs_table

def eval_score_lstm(song_names: list, songs: list, model, device):
    model.eval()
    with torch.no_grad():
        for idx, song, in enumerate(songs):
            song = song.float().to(device)
            output = model(song)
            print(f"{idx}\t{song_names[idx]}\t{output.item():.2f}")