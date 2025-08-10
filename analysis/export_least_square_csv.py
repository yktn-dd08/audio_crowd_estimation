from analysis.model_training import *


def export_csv(input_folder_list, valid_folder_list, model_name, model_param, target, time_agg, model_folder):
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
                          'count': torch.exp(y[:, 0]).detach().numpy() - 1.0})
    vxy_df = pd.DataFrame({'power': vx_pow.detach().numpy(),
                           'power_log': torch.log(vx_pow).detach().numpy(),
                           'count_log': vy[:, 0].detach().numpy(),
                           'count': torch.exp(vy[:, 0]).detach().numpy() - 1.0})

    os.makedirs(model_folder, exist_ok=True)
    logger.info(f'Saved XY csv files: {model_folder}/train_xy.csv, valid_xy.csv')
    xy_df.to_csv(f'{model_folder}/train_xy.csv', index=False)
    vxy_df.to_csv(f'{model_folder}/valid_xy.csv', index=False)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--input-config-json', type=str)
    args = parser.parse_args()

    with open(args.input_config_json, 'r') as f:
        cf = json.load(f)
    cf = cf['train']
    assert cf['model_name'] in MODEL_LIST
    export_csv(
        input_folder_list=cf['input_folder_list'],
        valid_folder_list=cf['valid_folder_list'] if 'valid_folder_list' in cf.keys() else None,
        model_folder=cf['model_folder'],
        model_name=cf['model_name'],
        model_param=cf['model_param'],
        target=['count'] if 'target' not in cf.keys() else cf['target'],
        time_agg=cf['time_agg']
    )
