import pickle
import sys
import os
from models.Mtsfn import MtsfnModel
from utils.save_checkpoint import save_checkpoint
from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
import numpy as np
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from args import args
import matplotlib.pyplot as plt

def _get_data(flag):
    
    if flag == 'test' or flag == 'pred':
        args.data_path = 'test_set.csv'
    elif flag == 'val':
        args.data_path = 'validation_set.csv'
    else:
        args.data_path = 'train_set.csv'
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader
with open('norm_params.pickle', 'rb') as f:
    scalers = pickle.load(f)
    mean_ = scalers.mean_
    std_ = scalers.scale_


def predict(model):
    pred_data, pred_loader = _get_data(flag='pred')
    model.load_state_dict(torch.load('./checkpoints/96-336-mine/model_5_0.55031365.pth'))
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            # 模型的输出
            outputs = outputs[:, -args.pred_len:, f_dim:]
            # 真实值
            batch_y = batch_y[:, -args.pred_len:, f_dim:].cuda()
            # pred = outputs.detach().cpu().numpy()  # .squeeze()
            outputs = (outputs.detach().cpu() * std_) + mean_
            truths = (batch_y.detach().cpu() * std_) + mean_
            inputs = (batch_x.detach().cpu() * std_) + mean_
            print(inputs.shape)
            print(truths.shape)
            print(outputs.shape)
            break
    #         preds.append(pred)
    # preds = np.array(preds)
    
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    X = torch.cat((inputs, truths), dim=1)
    X = X.permute(2,1,0).reshape(7,args.pred_len+args.seq_len)
    outputs = outputs.permute(2, 1, 0).reshape(7,args.pred_len)
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(0, args.pred_len+args.seq_len)], X[6], label='GroundTruth', color='orange')
    plt.plot([i for i in range(args.seq_len, args.pred_len+args.seq_len)], outputs[6], label='Prediction', color='blue')
    plt.legend()
    plt.title('Oil Temperature')
    plt.xlabel('Time Steps')
    plt.ylabel('Oil Temperature')
    plt.savefig('comparision.png')
    return

if __name__ == '__main__':
    # 实例化模型
    model = MtsfnModel(
        args
    ).float().cuda()
    
    predict(model)