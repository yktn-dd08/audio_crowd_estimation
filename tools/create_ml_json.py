import glob
import json
import argparse
import itertools
import os.path


DEFAULT_PARAM = {
    'SimpleCNN2': {
        'layer_num': 3,
        'kernel_size': [9, 23, 23],
        'dilation_size': [3, 1, 1],
        'inter_ch': [16, 13, 12],
        'pool_size': [2, 1, 1],
        'lr': 1e-05,
        'weight_decay': 0.0
    },
    'Conv1dTransformer': {
        'kernel_size': 1,
        'dilation_size': 1,
        'pool_size': 1,
        'token_dim': 6,
        'n_head': 3,
        'drop_out': 0.1,
        'layer_num': 3,
        'pe_flag': True,
        'pool_type': 'attention',
        'dff_times': 4,
        'feat_num': 64,
        'lr': 0.000001,
        'weight_decay': 0.0
    },
    'LeastSquareModel': {
        'lr': 0.000001,
        'weight_decay': 0.0
    }
}


def create_training_json(output_json, input_glob_list, valid_glob_list, model_folder, model_name, model_param):
    """
    モデル学習用のJSONファイルを作成する関数
    Parameters
    ----------
    output_json: str
        出力するJSONファイルのパス
    input_glob_list: list
        学習用データのglobパターンのリスト
    valid_glob_list: list
        検証用データのglobパターンのリスト
    model_folder: str
        モデルの保存先フォルダのパス
    model_name: str
        使用するモデルの名前 (例: 'SimpleCNN2', 'Transformer', 'LeastSquareModel')
    model_param: dict
        モデル学習に必要なパラメータを含む辞書。以下のキーを含む必要があります:
        - 'time_sec': float, 入力データの時間長さ (秒)
        - 'frame_num': int, 入力データのフレーム数
        - 'freq_num': int, 入力データの周波数数
        - 'target': list, 予測対象のラベルのリスト
        - 'model_param': dict, モデル固有のパラメータを含む辞書 (例: 'layer_num', 'kernel_size', 'dilation_size', 'inter_ch', 'pool_size', 'lr', 'weight_decay' など)
        - 'option': str, 学習オプション (例: 'train', 'test', 'predict' など)
        - 'batch_size': int, バッチサイズ (デフォルト: 64)
        - 'epoch': int, エポック数 (デフォルト: 200)
        - 'log_scale': bool, ログスケールを使用するかどうか (デフォルト: True)
        - 'time_agg': bool, 時間方向の集約を使用するかどうか (デフォルト: True)

    Returns
    -------

    """
    dev = model_param.get('device', None)
    target = model_param['target']
    batch_size = model_param.get('batch_size', 64)
    epoch = model_param.get('epoch', 200)
    log_scale = model_param.get('log_scale', True)
    time_agg = model_param.get('time_agg', True)
    if model_name not in DEFAULT_PARAM:
        raise ValueError(f'Model name "{model_name}" is not supported. Supported models: {list(DEFAULT_PARAM.keys())}')
    tf_param = {
        'time_sec': model_param['time_sec'],
        'frame_num': model_param['frame_num'],
        'freq_num': model_param['freq_num'],
        'out_features': len(model_param['target'])
    }
    def_param = DEFAULT_PARAM[model_name]
    # Merge default parameters with provided parameters
    merged_param = {**def_param, **model_param.get('model_param', {}), **tf_param}
    input_folder_list = []
    for igl in input_glob_list:
        input_folder_list.extend(glob.glob(igl))
    valid_folder_list = []
    for vgl in valid_glob_list:
        valid_folder_list.extend(glob.glob(vgl))
    opt = model_param['option']
    result_json = {
        'option': opt,
        'device': dev,
        opt: {
            'input_folder_list': input_folder_list,
            'valid_folder_list': valid_folder_list,
            'model_folder': model_folder,
            'model_name': model_name,
            'target': target,
            'batch_size': batch_size,
            'model_param': merged_param,
            'epoch': epoch,
            'log_scale': log_scale,
            'time_agg': time_agg
        }
    }
    output_folder = os.path.dirname(output_json)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_json, 'w') as f:
        json.dump(result_json, f, indent=4)
    return
