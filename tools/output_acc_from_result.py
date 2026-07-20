import argparse
import glob
import os.path

from analysis.model_common import *
from common.logger import get_logger

logger = get_logger('tools.output_acc_from_result')

def read_filtered_result(folder, rng):
    df = pd.read_csv(f'{folder}/test_result.csv')
    target_columns = [c.replace('target_', '', 1) for c in df.columns if c.startswith('target_')]
    result = {}
    for target_col in target_columns:
        p_df = df[[f'target_{target_col}', f'predict_{target_col}']].copy()
        p_df.columns = ['target', 'predict']
        p_df = p_df[(p_df['target'] >= rng[0]) & (p_df['target'] <= rng[1])].reset_index(drop=True)
        result[target_col] = p_df
    return result


def output_accuracy_all(glob_path, rng):
    folder_list = glob.glob(f'{glob_path}/*')
    folder_list = [fl for fl in folder_list if os.path.isdir(fl)]
    for fl in folder_list:
        logger.info(f'Calculating target value {rng[0]} - {rng[1]}: {fl}')
        filtered_dict = read_filtered_result(fl, rng)
        for col, c_df in filtered_dict.items():
            write_result(
                folder=fl,
                target_np=c_df[['target']].values,
                output_np=c_df[['predict']].values,
                target=[f'{col}_range{rng[0]}-{rng[1]}'],
                label='test',
                log_scale=False,
            )
            plot_result(
                folder=fl,
                target_np=c_df[['target']].values,
                output_np=c_df[['predict']].values,
                target=[f'{col}_range{rng[0]}-{rng[1]}'],
                label='test',
                log_scale=False,
            )
    return


def main():
    parser = argparse.ArgumentParser(description='Output Accuracy JSON files for ranged target values.')
    parser.add_argument('-g', '--glob-path', type=str)
    parser.add_argument('-r', '--range', type=float, nargs=2, default=[3.0, 8.0])
    args = parser.parse_args()
    output_accuracy_all(args.glob_path, args.range)
    return


if __name__ == '__main__':
    main()
