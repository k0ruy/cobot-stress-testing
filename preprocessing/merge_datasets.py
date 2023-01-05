# Libraries:
from pathlib import Path

import pandas as pd
import os
from config import SAVED_DATA


def merge_all_patients_csv_files():
    """
    Merge all the csv files of all patients by task into one csv file
    """
    # Supported tasks:
    tasks = ['cobot', 'manual']
    signals = ['ecg', 'eda', 'emg']

    for task in tasks:
        for signal in signals:
            # Get all the csv files of the current task and signal:
            csv_files = list((SAVED_DATA.rglob(f'{signal}_{task}*.csv')))
            print(csv_files)

            # Create a list of dataframes and concatenate them:
            dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]

            df = pd.concat(dataframes)
            # Save the merged dataframe:
            df.to_csv(SAVED_DATA / f'{signal}.csv', index=False)


if __name__ == '__main__':
    merge_all_patients_csv_files()
