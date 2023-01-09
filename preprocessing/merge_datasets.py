# Libraries:
from pathlib import Path
from time import sleep

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
            # Get all the csv files of the current task and signal ending with an _digit:
            csv_files = list((SAVED_DATA.rglob(f'{signal}_{task}_*.csv')))

            # Create a list of dataframes and concatenate them:
            dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]

            df = pd.concat(dataframes)
            # Save the merged dataframe:
            df.to_csv(SAVED_DATA / f'{signal}_{task}.csv', index=False)

    for task in ['rest', 'stroopeasy', 'stroophard']:
        # get all files in the patient folders with rest in the name; merge them into one csv file:
        files = list(SAVED_DATA.rglob(f'*{task}*.csv'))

        ecg_rest_files = [file for file in files if 'ecg' in str(file)]
        print(ecg_rest_files)
        eda_rest_files = [file for file in files if 'eda' in str(file)]
        print(eda_rest_files)
        emg_rest_files = [file for file in files if 'emg' in str(file)]
        print(emg_rest_files)

        # concatenate the rest files:
        ecg_rest_df = pd.concat([pd.read_csv(file) for file in ecg_rest_files])
        eda_rest_df = pd.concat([pd.read_csv(file) for file in eda_rest_files])
        emg_rest_df = pd.concat([pd.read_csv(file) for file in emg_rest_files])

        # save the rest files:
        ecg_rest_df.to_csv(SAVED_DATA / f'ecg_{task}.csv', index=False)
        eda_rest_df.to_csv(SAVED_DATA / f'eda_{task}.csv', index=False)
        emg_rest_df.to_csv(SAVED_DATA / f'emg_{task}.csv', index=False)


if __name__ == '__main__':
    merge_all_patients_csv_files()
