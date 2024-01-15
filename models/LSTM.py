import torch
from torch import nn

class Lstm_Model(nn.Module):
    def __init__(self, input_size, pred_size, hidden_size = 32, num_layers = 1):
        super(Lstm_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, pred_size)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')  # 2 for bidirection
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_size).to('cuda:0')
        x = torch.permute(x, (0,2,1))
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.fc(out)
        out = torch.permute(out, (0,2,1))
        return out
