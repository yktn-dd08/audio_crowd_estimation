import torch
import torch.nn as nn
import numpy as np
from model.vgg_cnn import *
from preprocess.disco_data import read_merged_data


def read_disco_data(folder):
    data_list = read_merged_data(folder)
    y = np.array([d['crowd'] for d in data_list])
    x = np.array([np.stack([d['logmel'], d['h_logmel'], d['v_logmel'], d['p_logmel']]) for d in data_list])
    return torch.tensor(y), torch.tensor(x)


def vgg_training(input_folder, output_pth):
    y, x = read_disco_data(input_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg11(in_channel=4, batch_norm=False).to(device)

    return


if __name__ == '__main__':
    print()
