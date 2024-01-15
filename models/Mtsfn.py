import os
import sys
import torch.nn.functional as F
import torch
from models.LSTM import Lstm_Model
from models.PatchTST import PatchTST_Model

from models.Transformer import Transformer_Model
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
# Cell
from torch import nn

class MtsfnModel(nn.Module):
    def __init__(self, configs):
        
        super().__init__()
        
        # load parameters
        self.input_size = configs.seq_len
        self.pred_len = configs.pred_len
        
        self.lstm_layer = Lstm_Model(self.input_size, self.pred_len)
        self.transformer_layer = Transformer_Model(configs=configs)
        self.patchtst = PatchTST_Model(configs=configs)
        self.raw_weights = nn.Parameter(torch.tensor([0.0, 0.0, 1]))  # 可训练的权重
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):           # x: [Batch, Input length, Channel]
        weights = F.softmax(self.raw_weights, dim=0)
        patst_out = self.patchtst(x_enc)
        transformer_out = self.transformer_layer(x_enc, x_mark_enc, x_dec, x_mark_dec)
        lstm_out = self.lstm_layer(x_enc)
        # 加权合并输出
        combined_output = (weights[0] * lstm_out +
                           weights[1] * transformer_out +
                           weights[2] * patst_out)
        
        return combined_output