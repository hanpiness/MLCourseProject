import torch
import adabound
import os


def save_checkpoint(state, is_best, single=True, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if single:
        fold = ''
    else:
        fold = str(state['fold']) + '_'
    cur_name = 'checkpoint.pth.tar'
    filepath = os.path.join(checkpoint, fold + cur_name)
    curpath = os.path.join(checkpoint, fold + 'model_cur.pth')

    torch.save(state, filepath)
    torch.save(state['state_dict'], curpath)

    if is_best:
        model_name = 'model_' + str(state['epoch']) + '_' + str(state['avg_val_loss']) + '.pth'
        model_path = os.path.join(checkpoint, fold + model_name)
        torch.save(state['state_dict'], model_path)