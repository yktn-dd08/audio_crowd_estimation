import argparse
import glob
import os
import re
import json
import pandas as pd
import numpy as np


def read_folder(folder_path):
    """
    Reads all CSV files in the specified folder and returns a dictionary of DataFrames.

    Parameters
    ----------
        folder_path (str): The path to the folder containing CSV files.
    """
    df_list = []
    prefix_list = ['train_acc', 'test_acc']
    for prefix in prefix_list:
        base_name = os.path.basename(folder_path)
        base_list = base_name.split('_')
        velocity = [bl for bl in base_list if re.fullmatch(r"v\d+", bl)]
        person = [bl for bl in base_list if re.fullmatch(r"p\d+", bl)]
        base_dict = {
            'tag': f'{base_list[0]}_{base_list[1]}',
            'place': base_list[0],
            'algorithm': base_list[1],
            'vel': velocity[0] if len(velocity) > 0 else '',
            'person': person[0] if len(person) > 0 else '',
        }
        json_list = glob.glob(os.path.join(folder_path, f"{prefix}_*.json"))
        for json_path in json_list:
            postfix = os.path.basename(json_path).split(f"{prefix}_")[1].split('.json')[0]
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame({'indicator': list(data.keys()), 'value': list(data.values())})
            df['condition'] = prefix.replace('_acc', '')
            df['target'] = postfix
            df['json_path'] = json_path
            for key, value in base_dict.items():
                df[key] = value
            df_list.append(df)
    dataframes = pd.concat(df_list, ignore_index=True)
    return dataframes


def merge_result(folder_list, output_path):
    """
    モデル学習時の出力結果をまとめる関数
    Parameters
    ----------
    folder_list: list
        list of folder paths containing the JSON files to be merged
    output_path: str
        path to the output CSV file where the merged results will be saved

    Returns
    -------

    """
    merge_df = pd.concat([read_folder(folder) for folder in folder_list], ignore_index=True)
    dir_name = os.path.dirname(output_path)
    os.makedirs(dir_name, exist_ok=True)
    merge_df.to_csv(output_path, index=False)
    return


def merge_result2(glob_path, output_path):
    folder_list = glob.glob(f'{glob_path}/*')
    folder_list = [fl for fl in folder_list if os.path.isdir(fl)]
    merge_result(folder_list, output_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSON files from multiple folders into a single CSV file.')
    parser.add_argument('-g', '--glob-path', type=str)
    parser.add_argument('-i', '--input_json', type=str)
    # parser.add_argument('-f', '--folder_list', nargs='+', help='List of folder paths containing the JSON files to be merged')
    parser.add_argument('-o', '--output_path', type=str, help='Path to the output CSV file where the merged results will be saved')
    args = parser.parse_args()
    merge_result2(args.glob_path, args.output_path)
