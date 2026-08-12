import glob
import json
import argparse
import itertools
import os.path


def extract_param_from_codex(codex_dict, v, p, scenario_num=12):
    search_result = codex_dict['results']
    assert isinstance(search_result, list), "codex_dict['results'] should be a list"
    ext_sr = {}
    for sr in search_result:
        if sr['target_mean_speed'] == v and sr['person_num'] == p:
            ext_sr = sr
            break
    result_list = []
    for sn in range(scenario_num):
        pass
    return


def sfm_trj_json(codex_json, setting_param, output_json, v_list, p_list, scenario_num=12):
    """
    codexの探索結果をもとに、v_list, p_listの組み合わせで、SFM人流Sim用のJSONを生成する。
    Parameters
    ----------
    codex_json: str
        必須, codexの探索結果JSONファイルパス
    setting_param: dict
        基本パラメータで下記を格納
        roi_shp: str, 必須, ROIのシェープファイルパス
        wall_shp: str, 任意, 壁のシェープファイルパス
        output_folder: str, 必須, 出力フォルダパス
        tag: str, 任意, 人流シミュレーションのタグ(ex: atc_sfm, gc_sfm, rec_sfm)
    output_json: str
        必須, 出力JSONファイルパス
    v_list: list[float]
        必須, 速度のリスト
    p_list: list[int]
        必須, 人数のリスト
    scenario_num: int
        任意, 1つのv, pの組み合わせに関して生成するシナリオ数, デフォルトは12

    Returns
    -------

    """
    with open(codex_json, 'r') as f:
        codex_dict = json.load(f)

    result_dict = {
        'option': 'social_force',
        'param': {
            'simulation_time': 3600,
            'roi_shp': setting_param['roi_shp'],
            'wall_shp': setting_param.get('wall_shp', None),
            'goal_flag': True,
            'dt': 1.0
        }
    }
    base_tag = setting_param['tag']
    for v, p in itertools.product(v_list, p_list):
        vv = f'{int(v * 10):02d}'
        tag_name = f'{base_tag}_v{vv}_p{p}'
        idx = 1
    return


def rnd_trj_json(setting_param, v_list, p_list, output_json):
    """
    各v_list, p_listの組み合わせで、ランダム人流Sim用のJSONを生成する。
    vとpの組み合わせは、v_listとp_listの全組み合わせで生成される。
    一つのv, pの組み合わせに関して12種類のシナリオを生成する。
    Parameters
    ----------
    output_json: str
        必須, 出力JSONファイルパス
    setting_param: dict
        基本パラメータで下記を格納
        roi_shp: str, 必須, ROIのシェープファイルパス
        wall_shp: str, 任意, 壁のシェープファイルパス
        output_folder: str, 必須, 出力フォルダパス
        tag: str, 任意, 人流シミュレーションのタグ(ex: atc_rnd, gc_rnd, rec_rnd)
    v_list: list[float]
        必須, 速度のリスト
    p_list: list[int]
        必須, 人数のリスト

    Returns
    -------

    """
    result_dict = {
        'option': 'random',
        'param': {
            'start_time': 0,
            'end_time': 3600,
            'dt': 1.0,
            'roi_shp': setting_param['roi_shp']
        }
    }
    if 'wall_shp' in setting_param:
        result_dict['param']['wall_shp'] = setting_param['wall_shp']
    base_tag = setting_param['tag']
    dir_division = [8, 4, 2]
    v_sigma = [0.3, 0.2, 0.1, 0.0]
    task_list = {}
    for v, p in itertools.product(v_list, p_list):
        vv = f'{int(v * 10):02d}'
        tag_name = f'{base_tag}_v{vv}_p{p}'
        idx = 1
        for dd, vs in itertools.product(dir_division, v_sigma):
            task_name = f'{tag_name}_crowd0101_{idx:02d}'
            task_value = {
                'output_csv': f"{setting_param['output_folder']}/roi1_{task_name}.csv",
                'dir_division': dd,
                'v': v,
                'v_sigma': vs,
                'person_num': p,
                'datetime_str': f'2024-01-01 {idx:02d}:00:00'
            }
            task_list[task_name] = task_value
            idx += 1
    result_dict['task_list'] = task_list
    folder = os.path.dirname(output_json)
    os.makedirs(folder, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result_dict, f, indent=4)
    return


def audio_json(setting_param, output_json):
    """
    設定フォルダ内のすべての人流データに関して、音響シミュレーション用のJSONを生成する。
    Parameters
    ----------
    setting_param: dict
        基本パラメータで下記を格納
        glob_path: str, 必須, 人流CSVのglobパス
        roi_shp: str, 必須, ROIのシェープファイルパス
        mic_shp: str, 必須, マイクのシェープファイルパス
        snr: float, 任意, SNRの値
        height: float, 任意, 人の高さ(m)
        max_order: int, 任意, 音響シミュレーションの最大反射次数
        output_folder: str, 必須, 音響信号の出力フォルダパス
    output_json: str
        必須, 出力JSONファイルパス

    Returns
    -------

    """
    glob_path = setting_param['glob_path']
    roi_shp = setting_param['roi_shp']
    mic_shp = setting_param['mic_shp']
    snr = setting_param.get('snr', None)
    height = setting_param.get('height', 3.0)
    max_order = setting_param.get('max_order', 0)
    output_folder = setting_param['output_folder']
    task_list = {}
    for file_path in glob.glob(glob_path):
        base_name = os.path.basename(file_path).split('.')[0]
        output_path = f'{output_folder}/{base_name}'
        task_list[base_name] = {
            'crowd_csv': file_path,
            'output_folder': output_path,
            'params': {}
        }
    result_dict = {
        'roi_shp': roi_shp,
        'mic_shp': mic_shp,
        'snr': snr,
        'height': height,
        'max_order': max_order,
        'task_list': task_list
    }
    folder = os.path.dirname(output_json)
    os.makedirs(folder, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(result_dict, f, indent=4)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', '--option', type=str, choices=['sfm_trj', 'rnd_trj'], required=True)
    parser.add_argument('-c', '--codex-json', type=str, help='Input codex JSON file path')
    parser.add_argument('-o', '--output-json', type=str, required=True, help='Output JSON file path')
    parser.add_argument('-v', '--v-list', type=float, nargs='+', required=True, help='List of speeds')
    parser.add_argument('-p', '--p-list', type=int, nargs='+', required=True, help='List of person numbers')
    parser.add_argument('-s', '--setting-json', type=str, help='Setting JSON file path for rnd_json option')
    args = parser.parse_args()
    if args.option == 'sfm_trj':
        if not args.codex_json:
            raise ValueError("codex-json is required for codex2json option")
        sfm_trj_json(args.codex_json, args.output_json, args.v_list, args.p_list)
    elif args.option == 'rnd_trj':
        if not args.setting_json:
            raise ValueError("setting-json is required for rnd_json option")
        with open(args.setting_json, 'r') as f:
            setting_param = json.load(f)
        rnd_trj_json(setting_param, args.v_list, args.p_list, args.output_json)
    return


if __name__ == '__main__':
    main()
