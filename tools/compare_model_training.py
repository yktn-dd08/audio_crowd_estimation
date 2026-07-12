import argparse
import glob
import os
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
        json_list = glob.glob(os.path.join(folder_path, f"{prefix}_*.json"))
        for json_path in json_list:
            postfix = os.path.basename(json_path).split(f"{prefix}_")[1].split('.json')[0]
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame({'indicator': list(data.keys()), 'value': list(data.values())})
            df['condition'] = prefix.replace('_acc', '')
            df['target'] = postfix
            df['json_path'] = json_path
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSON files from multiple folders into a single CSV file.')
    parser.add_argument('-i', '--input_json', type=str)
    # parser.add_argument('-f', '--folder_list', nargs='+', help='List of folder paths containing the JSON files to be merged')
    # parser.add_argument('-o', '--output_path', required=True, help='Path to the output CSV file where the merged results will be saved')
    args = parser.parse_args()
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
        merge_result(input_data['folder_list'], input_data['output_path'])