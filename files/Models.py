import torch
import torch.nn as nn
import torch.nn.functional as F
class ProgCNN(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super(ProgCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size = 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 2 * 4, 512),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(64, self.out_channels),
        )
        torch.nn.init.xavier_uniform_(self.conv1[0].weight)
        torch.nn.init.xavier_uniform_(self.conv2[0].weight)
        torch.nn.init.xavier_uniform_(self.conv3[0].weight)
        torch.nn.init.xavier_uniform_(self.conv4[0].weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        return x
    
class ProgCNNOld(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, dropout: float = 0.2):
        super(ProgCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size = 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size = 2)
        )

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 3 * 5, 512),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(64, self.out_channels),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.sigmoid(self.fc4(x))
        return x