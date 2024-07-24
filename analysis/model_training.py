import os.path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from model.vgg_cnn import *
from tqdm import tqdm
from preprocess.disco_data import read_merged_data
from sklearn.model_selection import train_test_split


def read_disco_data(folder):
    data_list = read_merged_data(folder)
    y = np.array([d['crowd'] for d in data_list])
    x = np.array([np.stack([d['logmel'], d['h_logmel'], d['v_logmel'], d['p_logmel']]) for d in data_list])
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
        for batch_idx, (x, y) in enumerate(test_loader):
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


def vgg_training(input_folder, output_pth, epoch):
    y, x = read_disco_data(input_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = vgg11(in_channel=4, batch_norm=False).to(device)
    tr_idx, ts_idx = train_test_split(range(len(y)), test_size=0.2, random_state=0)
    train_dataset = torch.utils.data.TensorDataset(x[tr_idx].to(device), y[tr_idx].to(device))
    test_dataset = torch.utils.data.TensorDataset(x[ts_idx].to(device), y[ts_idx].to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, weight_decay=2.7e-09)
    criterion = nn.MSELoss()

    train_loss, test_loss = [], []
    for e in range(epoch):
        tr_loss_tmp = model_train(model, train_dataloader, criterion, optimizer, epoch)
        ts_loss_tmp = model_test(model, test_dataloader, criterion, epoch)
        train_loss.append(tr_loss_tmp)
        test_loss.append(ts_loss_tmp)

    folder = os.path.dirname(output_pth)
    if not os.path.exists(folder):
        os.makedirs(folder)
    view_loss(train_loss, test_loss, f'{folder}/loss.png')
    torch.save(model.state_dict(), output_pth)

    return


if __name__ == '__main__':
    print()
