import argparse
import importlib
import json
import time

from common.logger import get_logger


logger = get_logger('tools.pipeline')


def apply_env_param(param, env_dict):
    """
    JSON内のパラメータに対して、環境変数(同じくJSON内に設定している$で始まるパラメータ)を適用する。
    Parameters
    ----------
    param: dict
        JSON内のパラメータ
    env_dict: dict
        環境変数の辞書

    Returns
    -------
    dict
    """
    if env_dict is {}:
        return param
    for key, value in param.items():
        if isinstance(value, dict):
            param[key] = apply_env_param(value, env_dict)
        else:
            if isinstance(value, str):
                if '$' in value:
                    for env_key, env_value in env_dict.items():
                        value = value.replace(f'{env_key}', env_value)
                    param[key] = value
                else:
                    param[key] = value
            else:
                param[key] = value
    return param


def execute_pipeline(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    env_dict = config.get('env_param', {})
    for ppl in config.get('pipelines', []):
        if 'module' not in ppl or 'function' not in ppl:
            raise ValueError(f"Pipeline configuration missing 'module', 'function': {ppl}")
        module_name = ppl['module']
        function_name = ppl['function']
        func = getattr(importlib.import_module(module_name), function_name, None)
        if func is None:
            raise AttributeError(f"Module '{module_name}' does not have a function '{function_name}'")
        common_params = ppl.get('params', {})
        common_params = apply_env_param(common_params, env_dict)
        logger.info(f"========== Pipeline executing: {module_name}.{function_name} started ==========")
        __start_time = time.time()
        if 'task_list' in ppl:
            for task_name, task_params in ppl['task_list'].items():
                merged_params = {**common_params, **task_params}
                merged_params = apply_env_param(merged_params, env_dict)
                logger.info(f"[Task '{task_name}']: params={merged_params}")
                func(**merged_params)
        else:
            logger.info(f"params={common_params}")
            func(**common_params)
        __end_time = time.time()
        logger.info(f"========== Pipeline executing: {module_name}.{function_name} finished {__end_time - __start_time} sec. ==========")

    logger.info(f"Pipeline executing finished.")

def main():
    parser = argparse.ArgumentParser(description="Execute a pipeline configuration.")
    parser.add_argument('config_path', type=str, help="Path to the pipeline configuration file.")
    args = parser.parse_args()
    execute_pipeline(args.config_path)
    return

if __name__ == '__main__':
    main()
