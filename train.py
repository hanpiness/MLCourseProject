import random
import sys
import os
from models.Mtsfn import MtsfnModel
import torch.nn.functional as F
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


def vali(model, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].cuda()
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            loss = criterion(pred, true)
            total_loss.append(loss)
    total_loss = np.average(total_loss)
    model.train()
    return total_loss

def _get_data(flag):
    
    if flag == 'test' or flag == 'pred':
        args.data_path = 'test_set.csv'
    elif flag == 'val':
        args.data_path = 'validation_set.csv'
    else:
        args.data_path = 'train_set.csv'
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return 0.5 * F.mse_loss(x, y) + 0.5 * F.l1_loss(x, y)
        

if __name__ == "__main__":

    train_data, train_loader = _get_data(flag='train')
    val_data, val_loader = _get_data(flag='val')
    test_data, test_loader = _get_data(flag='test')

    # 实例化模型
    model = MtsfnModel(
        args
    ).float().cuda()
    # model.load_state_dict(torch.load('./checkpoints/96-336-mine/model_95_0.46271154.pth'))
    # 损失函数和优化器
    criterion = CustomLoss()  # MSE损失
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_steps = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
                                    steps_per_epoch = train_steps,
                                    pct_start = args.pct_start,
                                    epochs = args.train_epochs,
                                    max_lr = args.learning_rate)
    best_loss = 1e10
    total_train_loss = []
    total_vail_loss = []
    random.seed(4)
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            optimizer.zero_grad()
            batch_x = batch_x.float().cuda()
            batch_y = batch_y.float().cuda()
            batch_x_mark = batch_x_mark.float().cuda()
            batch_y_mark = batch_y_mark.float().cuda()
            # 构造预热数据
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().cuda()
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].cuda()

            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                iter_count = 0
            loss.backward()
            optimizer.step()

        train_loss = np.average(train_loss)
        vali_loss = vali(model, val_data, val_loader, criterion)   # exchange
        test_loss = vali(model, test_data, test_loader, criterion)
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss, test_loss))

        if args.lradj != 'TST':
            adjust_learning_rate(optimizer, scheduler, epoch + 1, args)
        else:
            print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            
        # save_model
        is_best = vali_loss < best_loss
        best_loss = min(vali_loss, best_loss)
        save_checkpoint({
                        'fold': 0,
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'avg_train_loss':train_loss,
                        'avg_val_loss': vali_loss,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, single=True, checkpoint=args.checkpoint)
        total_train_loss.append(train_loss)
        total_vail_loss.append(vali_loss)
    epochs = range(1, len(total_train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, total_vail_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./loss.png')
    plt.show()