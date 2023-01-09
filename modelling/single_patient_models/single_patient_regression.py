# Library to classify stress for the cobot and manual experiments
# Using the random forest classifier, since it was the best performing model on the merged data set
# Data Manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.model_selection import train_test_split
from modelling.regression.rf_stress_prediction import handle_missing_values

# Global variables
from config import SAVED_DATA, COBOT_RESULTS, MANUAL_RESULTS

# Functions:
def load_patient_data(patient_id: int, task: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Function to load the ecg data for one of the 10 patients.
    @param patient_id: the patient id
    @param task: the task the patient was performing when the ecg data was collected
    :return: data: pd.DataFrame: the ecg, eda and emg data for the patient and task.
    """
    ecg = pd.read_csv(SAVED_DATA / f'patient-{patient_id:02d}' / f'ecg_{task}_{patient_id}.csv')
    eda = pd.read_csv(SAVED_DATA / f'patient-{patient_id:02d}' / f'eda_{task}_{patient_id}.csv')
    emg = pd.read_csv(SAVED_DATA / f'patient-{patient_id:02d}' / f'emg_{task}_{patient_id}.csv')

    return ecg, eda, emg


def split_patient_data(data: pd.DataFrame, add_demographics: bool = False) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Function to split the data into train and test set.
    @param data: the data to split
    @param add_demographics: whether to add the demographics to the data
    :return: x_train: pd.DataFrame: the training data
    :return: x_test: pd.DataFrame: the test data
    :return: y_train: pd.DataFrame: the training labels
    :return: y_test: pd.DataFrame: the test labels
    """
    columns_to_drop = ['patient', 'filename', 'Stress', 'NARS_S3', 'NARS_S2', 'NARS_S1',
                       'PSS', 'NASA_Total', 'Frustration', 'Physical_Demand', 'Mental_Demand', 'Fatigue', 'STAI_Total',
                       'File_number', 'Task', 'ID'] if add_demographics else \
        ['Age', 'Experience', 'Sex', 'patient', 'filename', 'Stress', 'NARS_S3', 'NARS_S2', 'NARS_S1', 'PSS',
         'NASA_Total', 'Frustration', 'Physical_Demand', 'Mental_Demand', 'Fatigue', 'STAI_Total', 'File_number',
         'Task', 'ID']

    x = data.drop(columns=columns_to_drop)
    y = data['Stress']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test


def main():
    """
    Main function to run the single patient regression random forest regressor models
    for the cobot and manual experiments.
    :return: None. Saves the results to the results folder.
    """
    for task in ['cobot', 'manual']:
        # get the correct path for the task:
        path = COBOT_RESULTS if task == 'cobot' else MANUAL_RESULTS

        for patient_nr in range(1, 11):
            # load the data
            ecg, eda, emg = load_patient_data(patient_nr, task)
            # handle missing values
            ecg, eda, emg = handle_missing_values(ecg), handle_missing_values(eda), handle_missing_values(emg)
            # split data, demographics are constant, so it's pointless to add them.
            x_train_ecg, x_test_ecg, y_train_ecg, y_test_ecg = split_patient_data(ecg)
            x_train_eda, x_test_eda, y_train_eda, y_test_eda = split_patient_data(eda)
            x_train_emg, x_test_emg, y_train_emg, y_test_emg = split_patient_data(emg)

            # get the merged data.
            x_train = pd.concat([x_train_ecg, x_train_eda, x_train_emg], axis=1)
            x_test = pd.concat([x_test_ecg, x_test_eda, x_test_emg], axis=1)
            y_train = y_train_ecg
            y_test = y_test_ecg

            # train the models
            rf_ecg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_eda = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_emg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_merged = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_ecg.fit(x_train_ecg, y_train_ecg)
            rf_eda.fit(x_train_eda, y_train_eda)
            rf_emg.fit(x_train_emg, y_train_emg)
            rf_merged.fit(x_train, y_train)
            # evaluate the models
            r2_ecg = rf_ecg.score(x_test_ecg, y_test_ecg)
            r2_eda = rf_eda.score(x_test_eda, y_test_eda)
            r2_emg = rf_emg.score(x_test_emg, y_test_emg)
            r2_merged = rf_merged.score(x_test, y_test)

            # save the results:
            patient_path = Path(path, f'patient-{patient_nr:02d}')
            patient_path.mkdir(parents=True, exist_ok=True)

            with open(patient_path / f'rf_stress_{task}.txt', 'w') as f:
                f.write(f'Patient {patient_nr} {task} results:\n')
                f.write(f'ECG R2: {r2_ecg:.3f}\n')
                f.write(f'EDA R2: {r2_eda:.3f}\n')
                f.write(f'EMG R2: {r2_emg:.3f}\n')
                f.write(f'Merged R2: {r2_merged:.3f}\n')


if __name__ == '__main__':
    main()
