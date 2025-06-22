import enum
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm


def get_device(device=None):
    if device is not None:
        return torch.device(device)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return dev


def model_train(model, train_loader, criterion, optimizer, epoch, verbose=True):
    model.train()
    train_loss = 0
    train_loss_tmp = 0
    with tqdm(train_loader, disable=not verbose) as _train_loader:
        for batch_idx, (x, y) in enumerate(_train_loader):
            _train_loader.set_description(f'[Epoch {epoch:03} - TRAIN]')
            _train_loader.set_postfix(LOSS=train_loss_tmp, LOSS_SUM=train_loss / len(train_loader.dataset))

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            train_loss_tmp = loss.item()
            train_loss += train_loss_tmp
            optimizer.step()

    return train_loss / len(train_loader.dataset)


def model_train_multitask(model, train_loader, criterion, task_criterion, optimizer, epoch, weight=0.5, verbose=True):
    model.train()
    train_loss, train_loss_tmp = 0, 0
    main_loss, main_loss_tmp = 0, 0
    task_loss, task_loss_tmp = 0, 0
    with tqdm(train_loader, disable=not verbose) as _train_loader:
        for batch_idx, (x, y, task) in enumerate(_train_loader):
            _train_loader.set_description(f'[Epoch {epoch:03} - TRAIN]')
            _train_loader.set_postfix(LOSS=train_loss_tmp, LOSS_SUM=train_loss / len(train_loader.dataset))
            
            optimizer.zero_grad()
            output, task_output = model(x)
            loss_main = criterion(output, y)
            loss_task = task_criterion(task_output, task)
            loss = loss_main + weight * loss_task
            loss.backward()
            main_loss_tmp = loss_main.item()
            main_loss += main_loss_tmp
            task_loss_tmp = loss_task.item()
            task_loss += task_loss_tmp
            train_loss_tmp = loss.item()
            train_loss += train_loss_tmp
            optimizer.step()

    return (train_loss / len(train_loader.dataset), main_loss / len(train_loader.dataset),
            task_loss / len(train_loader.dataset))


def model_test(model, test_loader, criterion, epoch, verbose=True):
    model.eval()
    test_loss = 0
    test_loss_tmp = 0
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                _test_loader.set_description(f'[Epoch {epoch:03} - TEST]')
                _test_loader.set_postfix(LOSS=test_loss_tmp, LOSS_SUM=test_loss / len(test_loader.dataset))
                output = model(x)
                loss = criterion(output, y)
                test_loss_tmp = loss.item()
                test_loss += test_loss_tmp
    return test_loss / len(test_loader.dataset)


def model_test_multitask(model, test_loader, criterion, task_criterion, epoch, weight=0.5, verbose=True):
    model.eval()
    total_loss, total_loss_tmp = 0, 0
    test_loss, test_loss_tmp = 0, 0
    task_loss, task_loss_tmp = 0, 0
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y, task) in enumerate(_test_loader):
                _test_loader.set_description(f'[Epoch {epoch:03} - TEST]')
                _test_loader.set_postfix(LOSS=total_loss_tmp, LOSS_SUM=total_loss / len(test_loader.dataset))
                output, task_output = model(x)
                loss_main = criterion(output, y)
                loss_task = task_criterion(task_output, task)
                loss = loss_main + weight * loss_task
                test_loss_tmp = loss_main.item()
                test_loss += test_loss_tmp
                task_loss_tmp = loss_task.item()
                task_loss += task_loss_tmp
                total_loss_tmp = loss.item()
                total_loss += total_loss_tmp
    return total_loss / len(test_loader.dataset), test_loss / len(test_loader.dataset), task_loss / len(test_loader.dataset)


def model_predict(model, test_loader, verbose=True):
    model.eval()
    target_list = []
    output_list = []
    # target_np = np.empty([0, 1])
    # output_np = np.empty([0, 1])
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                output = model(x)
                target_list.append(y.to('cpu').detach().numpy())
                output_list.append(output.to('cpu').detach().numpy())
                # target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                # output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    target_np = np.concatenate(target_list, axis=0)
    output_np = np.concatenate(output_list, axis=0)
    return target_np, output_np


def model_predict_multitask(model, test_loader, verbose=True):
    model.eval()
    target_list = []
    output_list = []
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y, task) in enumerate(_test_loader):
                output, _ = model(x)
                target_list.append(y.to('cpu').detach().numpy())
                output_list.append(output.to('cpu').detach().numpy())
                # target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                # output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    target_np = np.concatenate(target_list, axis=0)
    output_np = np.concatenate(output_list, axis=0)
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
    plt.close()
    return


def view_multi_loss(multi_loss_list, label_list, filename):
    plt.figure()
    assert len(multi_loss_list) == len(label_list)
    x = np.arange(1, len(multi_loss_list[0]) + 1)
    for label, loss in zip(label_list, multi_loss_list):
        plt.plot(x, loss, label=label)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.xlim((0, len(multi_loss_list[0])))
    plt.tick_params(labelsize=14)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.cla()
    plt.close()
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
    plt.close()
    return


def calculate_accuracy(target_np, output_np, json_path):
    acc = {'corr': float(np.corrcoef(target_np, output_np)[0, 1]),
           'mae': float(mean_absolute_error(target_np, output_np)),
           'mse': float(mean_squared_error(target_np, output_np)),
           'mape': float(mean_absolute_percentage_error(target_np, output_np))}
    with open(json_path, 'w') as f:
        json.dump(acc, f, indent=4)
    return
