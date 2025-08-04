import pandas as pd
import torch

from analysis.model_training import *


def export_csv(input_folder_list, valid_folder_list, model_name, model_param, target, time_agg, output_folder):
    logger.info(f'Export CSV for Least Square Method - Columns: {target}')
    (x, y), (vx, vy) = load_dataset(
        input_folder_list=input_folder_list,
        valid_folder_list=valid_folder_list,
        model_name=model_name,
        model_param=model_param,
        time_agg=time_agg,
        target=target
    )
    x_pow = torch.exp(x).mean(axis=[1, 2])
    vx_pow = torch.exp(vx).mean(axis=[1, 2])
    xy_df = pd.DataFrame({'power': x_pow.detach().numpy(),
                          'power_log': torch.log(x_pow).detach().numpy(),
                          'count_log': y[:, 0].detach().numpy(),
                          'count': torch.pow(y[:, 0]).detach().numpy() - 1.0})
    vxy_df = pd.DataFrame({'power': vx_pow.detach().numpy(),
                           'power_log': torch.log(vx_pow).detach().numpy(),
                           'count_log': vy[:, 0].detach().numpy(),
                           'count': torch.pow(vy[:, 0]).detach().numpy() - 1.0})
    return


if __name__ == '__main__':

    pass
