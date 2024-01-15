import sys
import os
from models.Mtsfn import MtsfnModel
from data_provider.data_factory import data_provider
from utils.tools import adjust_learning_rate
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from args import args
import matplotlib.pyplot as plt
import random

def calculate_mae_mse(predictions, ground_truth):
    # 计算 MAE 和 MSE
    mae = np.mean(np.abs(predictions - ground_truth), axis=(0, 1))
    mse = np.mean((predictions - ground_truth) ** 2, axis=(0, 1))

    return mae, mse

def vali(model, vali_data, vali_loader):
    # 初始化用于存储每次实验结果的列表
    all_predictions = []  # 用于存储每次实验的预测结果
    average_feature_mae = 0
    average_feature_mse = 0
    overall_average_mae = 0
    overall_average_mse = 0
    model.load_state_dict(torch.load('./checkpoints/96-336-mine/model_5_0.55031365.pth'))
    model.eval()
    for j in range(5):
        
        random.seed(j)
        test_predictions = torch.Tensor()  # 初始化为空的张量
        cumulative_feature_mae = np.zeros(7)
        cumulative_feature_mse = np.zeros(7)
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
                # 模型的输出
                outputs = outputs[:, -args.pred_len:, f_dim:]
                # 真实值
                batch_y = batch_y[:, -args.pred_len:, f_dim:].cuda()
                if test_predictions.numel() == 0:
                    test_predictions = outputs.detach().cpu()  # 如果是第一次迭代，直接赋值
                else:
                    test_predictions = torch.cat((test_predictions, outputs.detach().cpu()), dim=0)  # 否则拼接张量
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                # 计算每个特征的 MAE 和 MSE
                feature_mae, feature_mse = calculate_mae_mse(pred, true)
                cumulative_feature_mae += feature_mae
                cumulative_feature_mse += feature_mse
            # 计算所有批次的平均 MAE 和 MSE
            average_feature_mae += cumulative_feature_mae / len(vali_loader)
            average_feature_mse += cumulative_feature_mse / len(vali_loader)
            # 计算整体的平均 MAE 和 MSE
            overall_average_mae += np.mean(cumulative_feature_mae / len(vali_loader))
            overall_average_mse += np.mean(cumulative_feature_mse / len(vali_loader))
            # 将此次实验的预测结果添加到总列表中
            all_predictions.append(test_predictions)
    average_feature_mae = average_feature_mae / 5.0
    average_feature_mse = average_feature_mse / 5.0
    overall_average_mae = overall_average_mae / 5.0
    overall_average_mse = overall_average_mse / 5.0
    # 计算标准差
    # 假设我们关注的是每个特征在整个验证集上的标准差
    feature_std = torch.stack(all_predictions)
    feature_std = torch.std(feature_std, dim=0)
    feature_std = torch.sum(feature_std, dim=1)
    feature_std = torch.mean(feature_std, dim=0)
    
    print('total_mae: {0}, total_mse: {1}'.format(overall_average_mae, overall_average_mse))
    print('mae: {0}, mse: {1}'.format(average_feature_mae, average_feature_mse))
    print('feature_std: {}'.format(feature_std))
    print('average_std: {}'.format(np.average(feature_std)))
    
    return overall_average_mae, overall_average_mse

def _get_data(flag):
    
    if flag == 'test' or flag == 'pred':
        args.data_path = 'test_set.csv'
    elif flag == 'val':
        args.data_path = 'validation_set.csv'
    else:
        args.data_path = 'train_set.csv'
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader


if __name__ == "__main__":

    train_data, train_loader = _get_data(flag='train')
    val_data, val_loader = _get_data(flag='val')
    test_data, test_loader = _get_data(flag='test')

    # 实例化模型
    model = MtsfnModel(
        args
    ).float().cuda()

    overall_average_mae, overall_average_mse = vali(model, test_data, test_loader)