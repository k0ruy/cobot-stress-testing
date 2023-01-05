# Libraries:
from pathlib import Path

import pandas as pd
import os
from config import SAVED_DATA


def merge_all_patients_csv_files():
    """
    Merge all the csv files of all patients by task into one csv file
    """
    # Get all the csv files, exclude the stress labels, recursively:
    csv_files = [os.path.join(root, f) for root, dirs, files in os.walk(SAVED_DATA) for f in files]

    # Get the tasks:
    tasks = ['cobot', 'manual', 'rest', 'stroopeasy', 'stroophard']

    for task in tasks:
        # get all the files with this task
        task_files = [f for f in csv_files if task in f]

        dataframes = [pd.read_csv(f) for f in task_files]

        # save the file
        dataframes[0].to_csv(os.path.join(SAVED_DATA, f'merged_{task}.csv'), index=False)


if __name__ == '__main__':
    merge_all_patients_csv_files()