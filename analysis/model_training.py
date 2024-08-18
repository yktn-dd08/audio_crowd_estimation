import json
import os.path
import pickle
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from model.vgg_cnn import *
from tqdm import tqdm
from preprocess.disco_data import read_merged_data
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


CH = ['l', 'h', 'v', 'p']


def read_disco_data(folder):
    data_list = read_merged_data(folder)
    y = np.array([d['crowd'] for d in data_list])
    y = np.log(y + 1)
    x = np.array([np.stack([d['logmel'], d['h_logmel'], d['v_logmel'], d['p_logmel']]) for d in data_list])
    x = (x - x.flatten().mean()) / x.flatten().std()
    return torch.FloatTensor(y[:, np.newaxis]), torch.FloatTensor(x)


def read_sim_data(folder):
    with open(f'{folder}/feature.pickle', 'rb') as f:
        feature = pickle.load(f)
    y = feature['y']
    x = feature['x']
    y = np.log(y + 1)
    x = (x - x.flatten().mean()) / x.flatten().std()
    return torch.FloatTensor(y[:, np.newaxis]), torch.FloatTensor(x)


def model_train(model, train_loader, criterion, optimizer, epoch, verbose=True):
    model.train()
    train_loss = 0
    train_loss_tmp = 0
    with tqdm(train_loader, disable=not verbose) as _train_loader:
        for batch_idx, (x, y) in enumerate(_train_loader):
            _train_loader.set_description(f'[Epoch {epoch:03} - TRAIN]')
            _train_loader.set_postfix(LOSS=train_loss_tmp, LOSS_SUM=train_loss/len(train_loader.dataset))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            train_loss_tmp = loss.item()
            train_loss += train_loss_tmp
            optimizer.step()

    return train_loss / len(train_loader.dataset)


def model_test(model, test_loader, criterion, epoch, verbose=True):
    model.eval()
    test_loss = 0
    test_loss_tmp = 0
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                _test_loader.set_description(f'[Epoch {epoch:03} - TEST]')
                _test_loader.set_postfix(LOSS=test_loss_tmp, LOSS_SUM=test_loss/len(test_loader.dataset))
                output = model(x)
                loss = criterion(output, y)
                test_loss_tmp = loss.item()
                test_loss += test_loss_tmp
    return test_loss / len(test_loader.dataset)


def model_predict(model, test_loader, verbose=True):
    model.eval()
    target_np = np.empty([0, 1])
    output_np = np.empty([0, 1])
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                output = model(x)
                target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    return target_np, output_np


def view_loss(train_loss, test_loss, filename):
    plt.figure()
    x = np.arange(1, len(train_loss) + 1)
    plt.plot(x, train_loss, label='TRAIN')
    plt.plot(x, test_loss, label='TEST')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xlim((0, len(train_loss)))
    plt.tick_params(labelsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.cla()
    return


def scatter_plot(target, output, filename):
    plt.figure(figsize=(10, 10))
    plt.scatter(target, output)
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.xlim([min(target.min(), output.min()), max(target.max(), output.max())])
    plt.ylim([min(target.min(), output.min()), max(target.max(), output.max())])
    plt.grid(True)
    plt.savefig(filename)
    plt.cla()
    return


def calculate_accuracy(target_np, output_np, json_path):
    acc = {'corr': np.corrcoef(target_np, output_np)[0, 1],
           'mae': mean_absolute_error(target_np, output_np),
           'mse': mean_squared_error(target_np, output_np),
           'mape': mean_absolute_percentage_error(target_np, output_np)}
    with open(json_path, 'w') as f:
        json.dump(acc, f)
    return


def vgg_training_cv(input_folder, output_folder, epoch, vgg=11, batch_norm=False, ch='lhvp', k=5, data='disco'):
    y, x = read_disco_data(input_folder) if data == 'disco' else read_sim_data(input_folder)
    x_ch_idx = [CH.index(c) for c in ch]
    sorted(x_ch_idx)
    x = x[:, x_ch_idx, :, :]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    target, output = [], []
    for cv, (tr_idx, ts_idx) in enumerate(kf.split(x, y)):
        model = None
        if vgg == 11:
            model = vgg11(in_channel=len(ch), batch_norm=batch_norm).to(device)
        elif vgg == 13:
            model = vgg13(in_channel=len(ch), batch_norm=batch_norm).to(device)
        elif vgg == 16:
            model = vgg16(in_channel=len(ch), batch_norm=batch_norm).to(device)
        elif vgg == 19:
            model = vgg19(in_channel=len(ch), batch_norm=batch_norm).to(device)
        else:
            Exception('input vgg=11, 13, 16 or 19.')

        train_dataset = torch.utils.data.TensorDataset(x[tr_idx].to(device), y[tr_idx].to(device))
        test_dataset = torch.utils.data.TensorDataset(x[ts_idx].to(device), y[ts_idx].to(device))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
        criterion = nn.MSELoss()

        print(f'[Cross-validation: {cv}]: # of train data = {len(tr_idx)}, # of test data = {len(ts_idx)}')
        train_loss, test_loss = [], []
        for e in range(epoch):
            tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, e)
            ts_loss_tmp = model_test(model, test_dataloader, criterion, e)
            train_loss.append(tr_loss_tmp)
            test_loss.append(ts_loss_tmp)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        view_loss(train_loss, test_loss, f'{output_folder}/loss_fold{cv}.png')
        torch.save(model.state_dict(), f'{output_folder}/model_fold{cv}.pth')

        target_np, output_np = model_predict(model, test_dataloader)
        scatter_plot(target_np[:, 0], output_np[:, 0], f'{output_folder}/scatter_log_fold{cv}.png')
        target_np = np.exp(target_np[:, 0]) - 1
        output_np = np.exp(output_np[:, 0]) - 1
        scatter_plot(target_np, output_np, f'{output_folder}/scatter_fold{cv}.png')
        calculate_accuracy(target_np, output_np, f'{output_folder}/accuracy_fold{cv}.json')
        target.append(target_np)
        output.append(output_np)

        target_np_close, output_np_close = model_predict(model, train_dataloader)
        scatter_plot(target_np_close[:, 0], output_np_close[:, 0],
                     f'{output_folder}/train_scatter_log_fold{cv}.png')
        target_np_close = np.exp(target_np_close[:, 0]) - 1
        output_np_close = np.exp(output_np_close[:, 0]) - 1
        scatter_plot(target_np_close, output_np_close, f'{output_folder}/train_scatter_fold{cv}.png')
        calculate_accuracy(target_np_close, output_np_close, f'{output_folder}/train_accuracy_fold{cv}.json')
    target = np.concatenate(target)
    output = np.concatenate(output)
    scatter_plot(target, output, f'{output_folder}/all_scatter.png')
    calculate_accuracy(target, output, f'{output_folder}/all_accuracy.json')
    return


def vgg_training(input_folder, output_folder, epoch, vgg=11, batch_norm=False, ch='lhvp', data='disco'):
    y, x = read_disco_data(input_folder) if data == 'disco' else read_sim_data(input_folder)
    x_ch_idx = [CH.index(c) for c in ch]
    sorted(x_ch_idx)
    x = x[:, x_ch_idx, :, :]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if vgg == 11:
        model = vgg11(in_channel=len(ch), batch_norm=batch_norm).to(device)
    elif vgg == 13:
        model = vgg13(in_channel=len(ch), batch_norm=batch_norm).to(device)
    elif vgg == 16:
        model = vgg16(in_channel=len(ch), batch_norm=batch_norm).to(device)
    elif vgg == 19:
        model = vgg19(in_channel=len(ch), batch_norm=batch_norm).to(device)
    else:
        Exception('input vgg=11, 13, 16 or 19.')

    tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    train_dataset = torch.utils.data.TensorDataset(x[tr_idx].to(device), y[tr_idx].to(device))
    test_dataset = torch.utils.data.TensorDataset(x[ts_idx].to(device), y[ts_idx].to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
    criterion = nn.MSELoss()

    train_loss, test_loss = [], []
    for e in range(epoch):
        tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, e)
        ts_loss_tmp = model_test(model, test_dataloader, criterion, e)
        train_loss.append(tr_loss_tmp)
        test_loss.append(ts_loss_tmp)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # folder = os.path.dirname()
    view_loss(train_loss, test_loss, f'{output_folder}/loss.png')
    torch.save(model.state_dict(), f'{output_folder}/model.pth')

    target_np, output_np = model_predict(model, test_dataloader)
    target_np = np.exp(target_np[:, 0]) - 1
    output_np = np.exp(output_np[:, 0]) - 1
    scatter_plot(target_np, output_np, f'{output_folder}/scatter.png')
    calculate_accuracy(target_np, output_np, f'{output_folder}/accuracy.json')

    target_np_close, output_np_close = model_predict(model, train_dataloader)
    target_np_close = np.exp(target_np_close[:, 0]) - 1
    output_np_close = np.exp(output_np_close[:, 0]) - 1
    scatter_plot(target_np_close, output_np_close, f'{output_folder}/scatter_train.png')
    calculate_accuracy(target_np_close, output_np_close, f'{output_folder}/accuracy_train.json')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, default='train', choices=['train', 'cv'])
    parser.add_argument('-i', '--input-folder', type=str)
    parser.add_argument('-o', '--output-folder', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-f', '--feature', type=str, default='lhvp')
    parser.add_argument('-v', '--vgg', type=int, default=11, choices=[11, 13, 16, 19])
    parser.add_argument('-b', '--batch-norm', type=str, default='False', choices=['True', 'False'])
    parser.add_argument('-d', '--data', type=str, default='disco', choices=['disco', 'sim'])
    args = parser.parse_args()
    if args.option == 'train':
        vgg_training(args.input_folder, args.output_folder, args.epoch, args.vgg, args.batch_norm == 'True',
                     args.feature, args.data)
    elif args.option == 'cv':
        vgg_training_cv(args.input_folder, args.output_folder, args.epoch, args.vgg, args.batch_norm == 'True',
                        args.feature, 5, args.data)

