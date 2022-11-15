# Libraries:
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from tabulate import tabulate
# random forest regressor:
from sklearn.ensemble import RandomForestRegressor

# Driver:
if __name__ == '__main__':

    # get the stress labels for each user:
    with open(Path('..', "questionnaires.pkl"), "rb") as f:
        subject_data = pickle.load(f)

        # group by ID and get the mean of the stress column:
        y = pd.DataFrame(subject_data.groupby('ID')['Stress'].mean())

    # load the pandas dataframe with the ecg features:
    df_ecg = pd.read_csv(Path('..', 'saved_data', 'ecg_time_and_freq_features_cobot.csv'))
    print(df_ecg.shape)




