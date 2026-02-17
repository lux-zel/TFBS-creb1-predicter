# model.py
import torch.nn as nn

class BindingCNN(nn.Module):
    def __init__(self, seq_len=200):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=12, padding=6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(3, stride=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(3, stride=2)

        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()