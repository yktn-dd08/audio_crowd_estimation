import glob
import argparse
import json
import os
import pickle

import lightgbm as lgb
import pandas as pd

from preprocess.disco_data import read_merged_data


def shap_analysis(folder):
    data_list = read_merged_data(folder)
    df = pd.DataFrame(data_list)
    df = df.select_dtypes(exclude='object')

    return


if __name__ == '__main__':
    print()

