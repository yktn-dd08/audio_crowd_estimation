import json
import math

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from sklearn.utils import deprecated
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm
from scipy import interpolate

_STFT_WINDOW_LENGTH_SECONDS = 0.025
_STFT_HOP_LENGTH_SECONDS = 0.010
_LOG_OFFSET = 1.0e-20
_MEL_MIN_HZ = 125
_MEL_MAX_HZ = 7500
_NUM_BANDS = 64

def get_device(device=None):
    """
    deviceの取得
    Parameters
    ----------
    device: str or None
        'cpu', 'cuda', 'mps' or None
        Noneの場合、自動で判定

    Returns
    -------
    dev: torch.device
        使用するdevice
    """
    if device is not None:
        return torch.device(device)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    return dev


def trans_logmel(signal: np.ndarray | torch.Tensor, fs=16000, astype_tensor=True):
    """
    音響信号から対数メルスペクトログラムを計算する
    Parameters
    ----------
    signal
    fs
    astype_tensor

    Returns
    -------

    """
    if isinstance(signal, np.ndarray):
        signal = torch.Tensor(signal)
    window_length_samples = int(round(fs * _STFT_WINDOW_LENGTH_SECONDS))
    hop_length_samples = int(round(fs * _STFT_HOP_LENGTH_SECONDS))
    fft_length = 2 ** int(math.ceil(math.log(window_length_samples) / math.log(2.0)))
    mel_spec = MelSpectrogram(
        sample_rate=fs,
        n_fft=fft_length,
        hop_length=hop_length_samples,
        f_min=_MEL_MIN_HZ,
        f_max=_MEL_MAX_HZ,
        n_mels=_NUM_BANDS
    )
    logmel = torch.log(mel_spec(signal) + _LOG_OFFSET)
    return logmel if astype_tensor else logmel.detach().numpy()


class SampleWeightCalculator:
    def __init__(self, y: np.ndarray | torch.Tensor, dev=None, sample_num=None, alpha=0.1, eps=1e-6):
        """
        目的変数の値の分布に基づいてサンプル重みを計算するクラス
        Parameters
        ----------
        y: np.array or torch.Tensor
            目的変数の値の配列
        dev: torch.device
            サンプル重みを格納するデバイス
        sample_num: int or None
            サンプル数。Noneの場合、全てのサンプルを使用する
        alpha: float
            重みの計算に使用する指数。値が大きいほど、頻度の低いサンプルに対して重みが大きくなる
        eps: float
            重みの計算に使用する小さな値。頻度が0のサンプルに対して重みが無限大になるのを防ぐために使用する
        """
        self.dev = dev
        if isinstance(y, torch.Tensor):
            y = y.to('cpu').detach().numpy()
        self.dim = y.shape[1]
        y_flat = y.flatten()
        if sample_num is not None:
            y_flat = np.random.choice(y_flat, size=sample_num, replace=False)

        # 頻度値を計算
        val, cnt = np.unique(y_flat, return_counts=True)
        # 目的変数の値と頻度の関係を二次補間して関数化
        self.func = interpolate.interp1d(val, cnt, kind='quadratic', bounds_error=False, fill_value='extrapolate')
        self.length = sum(cnt)
        self.alpha = alpha
        self.eps = eps

    def weight(self, y: np.ndarray | torch.Tensor):
        """
        サンプル重みを計算するメソッド
        Parameters
        ----------
        y: np.array or torch.Tensor
            サンプル重みを計算する目的変数の値の配列

        Returns
        -------
        weights: torch.Tensor
            計算されたサンプル重みのテンソル
        """
        if isinstance(y, torch.Tensor):
            y = y.to('cpu').detach().numpy()
        assert y.shape[1] == self.dim, f'Invalid dimension. y.shape: {y.shape}, expected: (*, {self.dim}).'
        weights = np.array([1.0 / np.prod((self.func(each_y) / self.length) ** self.alpha + self.eps) for each_y in y])
        if self.dev is not None:
            weights = torch.Tensor(weights).to(self.dev)
        return weights


def weighted_loss(output, target, sample_weight, criterion):
    if isinstance(criterion, torch.nn.MSELoss):
        per_elem_loss = F.mse_loss(output, target, reduction='none')
    elif isinstance(criterion, torch.nn.L1Loss):
        per_elem_loss = F.l1_loss(output, target, reduction='none')
    else:
        raise ValueError(f'Unsupported criterion: {type(criterion)}')

    per_sample_loss = per_elem_loss.mean(dim=1)
    loss = (per_sample_loss * sample_weight).sum() / (sample_weight.sum() + 1e-8)
    return loss


def model_train(model, train_loader, criterion, optimizer, epoch, dev=None, verbose=True):
    """
    pytorchモデル学習用　devがNoneの場合はGPU転送なし、devが指定されている場合はGPU転送ありのメソッド
    Parameters
    ----------
    model: torch.nn.Module
        学習するモデル
    train_loader: torch.utils.data.DataLoader
        学習データのDataLoader
    criterion: torch.nn.Module
        損失関数
    optimizer: torch.optim.Optimizer
        最適化手法
    epoch: int
        エポック数
    dev: torch.device or None
        使用するデバイス。Noneの場合はGPU転送なし、torch.deviceが指定
    verbose: bool
        進捗表示の有無

    Returns
    -------
    result: float
        学習データに対する平均損失値
    """
    model.train()
    train_loss = 0
    train_loss_tmp = 0
    total_samples = 0
    with tqdm(train_loader, disable=not verbose) as _train_loader:
        for batch_idx, (x, y) in enumerate(_train_loader):
            _train_loader.set_description(f'[Epoch {epoch:03} - TRAIN]')
            avg_loss = train_loss / max(total_samples, 1)
            _train_loader.set_postfix(LOSS=train_loss_tmp, LOSS_SUM=avg_loss)

            optimizer.zero_grad()
            output = model(x if dev is None else x.to(dev))
            loss = criterion(output, y if dev is None else y.to(dev))
            loss.backward()
            batch_size = y.size(0)
            train_loss_tmp = loss.item() * batch_size
            train_loss += train_loss_tmp
            total_samples += batch_size
            optimizer.step()
    result = train_loss / total_samples
    return result


def model_train_sw(
        model,
        train_loader,
        criterion,
        optimizer,
        epoch,
        sample_weight: SampleWeightCalculator,
        dev=None,
        verbose=True
):
    """
    pytorchモデル学習用　devがNoneの場合はGPU転送なし、devが指定されている場合はGPU転送ありのメソッド
    サンプル重みを使用して学習する
    Parameters
    ----------
    model: torch.nn.Module
        学習するモデル
    train_loader: torch.utils.data.DataLoader
        学習データのDataLoader
    criterion: torch.nn.Module
        損失関数
    optimizer: torch.optim.Optimizer
        最適化手法
    epoch: int
        エポック数
    sample_weight: SampleWeightCalculator
        サンプル重みを計算するクラスのインスタンス
    dev: torch.device
        使用するデバイス。Noneの場合はGPU転送なし、torch.deviceが指定
    verbose: bool
        進捗表示の有無

    Returns
    -------
    result: float
        学習データに対する平均損失値
    """
    model.train()
    train_loss = 0
    train_loss_tmp = 0
    total_weight_sum = 0.0
    with tqdm(train_loader, disable=not verbose) as _train_loader:
        for batch_idx, (x, y) in enumerate(_train_loader):
            _train_loader.set_description(f'[Epoch {epoch:03} - TRAIN]')
            avg_loss = train_loss / max(total_weight_sum, 1.0e-8)
            _train_loader.set_postfix(LOSS=train_loss_tmp, LOSS_SUM=avg_loss)

            optimizer.zero_grad()
            output = model(x if dev is None else x.to(dev))
            w = sample_weight.weight(y if dev is None else y.to(dev))
            # loss = criterion(output, y if dev is None else y.to(dev))
            loss = weighted_loss(output, y if dev is None else y.to(dev), w, criterion)
            loss.backward()
            batch_weight_sum = w.sum().item()
            train_loss_tmp = loss.item()
            train_loss += train_loss_tmp * batch_weight_sum
            total_weight_sum += batch_weight_sum
            optimizer.step()
    result = train_loss / max(total_weight_sum, 1.0e-8)
    return result


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


def model_test(model, test_loader, criterion, epoch, dev=None, verbose=True):
    """
    pytorchモデル評価用　devがNoneの場合はGPU転送なし、devが指定されている場合はGPU転送ありのメソッド
    Parameters
    ----------
    model: torch.nn.Module
        評価するモデル
    test_loader: torch.utils.data.DataLoader
        テストデータのDataLoader
    criterion: torch.nn.Module
        損失関数
    epoch: int
        エポック数
    dev: torch.device or None
        使用するデバイス。Noneの場合はGPU転送なし、torch.deviceが指定
    verbose: bool
        進捗表示の有無

    Returns
    -------
    result: float
        テストデータに対する平均損失値
    """
    model.eval()
    test_loss = 0
    test_loss_tmp = 0
    total_samples = 0
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                _test_loader.set_description(f'[Epoch {epoch:03} - TEST]')
                avg_loss = test_loss / max(total_samples, 1)
                _test_loader.set_postfix(LOSS=test_loss_tmp, LOSS_SUM=avg_loss)
                output = model(x if dev is None else x.to(dev))
                loss = criterion(output, y if dev is None else y.to(dev))
                batch_size = y.size(0)
                test_loss_tmp = loss.item() * batch_size
                test_loss += test_loss_tmp
                total_samples += batch_size
    result = test_loss / total_samples
    return result


def model_test_sw(
        model,
        test_loader,
        criterion,
        epoch,
        sample_weight: SampleWeightCalculator,
        dev=None,
        verbose=True
):
    """
    学習、評価時にGPU転送する場合のメソッド
    Parameters
    ----------
    model: torch.nn.Module
        学習するモデル
    test_loader: torch.utils.data.DataLoader
        テストデータのDataLoader
    criterion: torch.nn.Module
        損失関数
    epoch: int
        エポック数
    sample_weight: SampleWeightCalculator
        サンプル重みを計算するクラスのインスタンス
    dev: torch.device
        使用するデバイス
    verbose: bool
        進捗表示の有無

    Returns
    -------
    result: float
        テストデータに対する平均損失値
    """
    model.eval()
    test_loss = 0
    test_loss_tmp = 0
    total_weight_sum = 0.0
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                _test_loader.set_description(f'[Epoch {epoch:03} - TEST]')
                avg_loss = test_loss / max(total_weight_sum, 1.0e-8)
                _test_loader.set_postfix(LOSS=test_loss_tmp, LOSS_SUM=avg_loss)
                output = model(x if dev is None else x.to(dev))
                w = sample_weight.weight(y if dev is None else y.to(dev))
                # loss = criterion(output, y if dev is None else y.to(dev))
                loss = weighted_loss(output, y if dev is None else y.to(dev), w, criterion)
                batch_weight_sum = w.sum().item()
                test_loss_tmp = loss.item()
                test_loss += test_loss_tmp * batch_weight_sum
                total_weight_sum += batch_weight_sum
    result = test_loss / max(total_weight_sum, 1.0e-8)
    return result


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


def model_predict(model, test_loader, dev=None, verbose=True):
    """
    pytorchモデル予測用　devがNoneの場合はGPU転送なし、devが指定されている場合はGPU転送ありのメソッド
    Parameters
    ----------
    model: torch.nn.Module
        予測するモデル
    test_loader: torch.utils.data.DataLoader
        テストデータのDataLoader
    dev: torch.device or None
        使用するデバイス。Noneの場合はGPU転送なし、torch.deviceが指定
    verbose: bool
        進捗表示の有無

    Returns
    -------
    target_np: np.ndarray
        目的変数の値のnumpy配列
    output_np: np.ndarray
        予測値のnumpy配列
    """
    model.eval()
    target_list = []
    output_list = []
    # target_np = np.empty([0, 1])
    # output_np = np.empty([0, 1])
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                output = model(x if dev is None else x.to(dev))
                target_list.append(y.to('cpu').detach().numpy())
                output_list.append(output.to('cpu').detach().numpy())
                # target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                # output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    target_np = np.concatenate(target_list, axis=0)
    output_np = np.concatenate(output_list, axis=0)
    return target_np, output_np


# @deprecated("model_predict_wg is deprecated. Use model_predict instead.")
def model_predict_wg(model, test_loader, dev, verbose=True):
    model.eval()
    target_list = []
    output_list = []
    # target_np = np.empty([0, 1])
    # output_np = np.empty([0, 1])
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y) in enumerate(_test_loader):
                output = model(x.to(dev))
                target_list.append(y.to('cpu').detach().numpy())
                output_list.append(output.to('cpu').detach().numpy())
                # target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                # output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    target_np = np.concatenate(target_list, axis=0)
    output_np = np.concatenate(output_list, axis=0)
    return target_np, output_np


def model_predict_multitask(model, test_loader, verbose=True):
    model.eval()
    target_list, target_task_list = [], []
    output_list, output_task_list = [], []
    with torch.no_grad():
        with tqdm(test_loader, disable=not verbose) as _test_loader:
            for batch_idx, (x, y, task) in enumerate(_test_loader):
                output, output_task = model(x)
                target_list.append(y.to('cpu').detach().numpy())
                output_list.append(output.to('cpu').detach().numpy())
                target_task_list.append(task.to('cpu').detach().numpy())
                output_task_list.append(output_task.to('cpu').detach().numpy())
                # target_np = np.concatenate([target_np, y.to('cpu').detach().numpy()], axis=0)
                # output_np = np.concatenate([output_np, output.to('cpu').detach().numpy()], axis=0)
    target_np = np.concatenate(target_list, axis=0)
    output_np = np.concatenate(output_list, axis=0)
    target_task_np = np.concatenate(target_task_list, axis=0)
    output_task_np = np.concatenate(output_task_list, axis=0)
    return target_np, output_np, target_task_np, output_task_np


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
    acc = {'corr': float(np.corrcoef(target_np, output_np)[1, 0]),
           'mae': float(mean_absolute_error(target_np, output_np)),
           'mse': float(mean_squared_error(target_np, output_np)),
           'rmse': float(mean_squared_error(target_np, output_np)) ** 0.5,
           'mape': float(mean_absolute_percentage_error(target_np, output_np))}
    with open(json_path, 'w') as f:
        json.dump(acc, f, indent=4)
    return


def write_result(folder, target_np, output_np, target, label, log_scale):
    """
    解析結果の保存
    Parameters
    ----------
    folder: str
        保存先フォルダ
    target_np: np.ndarray
        目的変数のnumpy配列
    output_np: np.ndarray
        予測値のnumpy配列
    target: list
        目的変数のカラム名リスト
    label: str
        保存ファイル名のラベル
    log_scale: bool
        目的変数が対数変換されているかどうか

    Returns
    -------

    """
    if log_scale:
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    target_df = pd.DataFrame(target_np)
    target_df.columns = [f'target_{t}' for t in target]
    output_df = pd.DataFrame(output_np)
    output_df.columns = [f'predict_{t}' for t in target]
    res_df = pd.concat([target_df, output_df], axis=1)
    res_df.to_csv(f'{folder}/{label}_result.csv', index=False)
    for t in target:
        calculate_accuracy(target_df[f'target_{t}'].values, output_df[f'predict_{t}'].values,
                           f'{folder}/{label}_acc_{t}.json')
    return


def plot_result(folder, target_np, output_np, target: list, label, log_scale):
    """
    解析結果のプロット保存
    Parameters
    ----------
    folder: str
        保存先フォルダ
    target_np: np.ndarray
        目的変数のnumpy配列
    output_np: np.ndarray
        予測値のnumpy配列
    target: list
        目的変数のカラム名リスト
    label: str
        保存ファイル名のラベル
    log_scale: bool
        目的変数が対数変換されているかどうか

    Returns
    -------

    """
    assert len(target) == target_np.shape[1], f'Invalid size. target: {target}, target_np.shape: {target_np.shape}.'
    if log_scale:
        for i, t in enumerate(target):
            scatter_plot(target_np[:, i], output_np[:, i], f'{folder}/{label}_scatter_log_{t}.png')
        target_np = np.exp(target_np) - 1
        output_np = np.exp(output_np) - 1
    for i, t in enumerate(target):
        scatter_plot(target_np[:, i], output_np[:, i], f'{folder}/{label}_scatter_{t}.png')
    return
