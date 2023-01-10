# Library to perform lasso regression on each patient's data:
# Data manipulation:
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from modelling.regression.rf_stress_prediction import handle_missing_values
# Modelling:
from single_patient_rf_regression import load_patient_data, split_patient_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso


# Global variables:
from config import RESULTS
mc_path = Path(RESULTS / 'merged_results' / 'lasso_regression')
mc_path.mkdir(parents=True, exist_ok=True)
m_path = Path(RESULTS / 'manual_results' / 'lasso_regression')
m_path.mkdir(parents=True, exist_ok=True)
c_path = Path(RESULTS / 'cobot_results' / 'lasso_regression')
c_path.mkdir(parents=True, exist_ok=True)


def lasso_regression(patient_data: pd.DataFrame, alpha: float = 0.1, max_iter: int = 1000,
                     dataset_folder: str = "cobot_results", patient_id: int = 1, type: str = "ecg") -> pd.DataFrame:
    """
    Perform lasso regression on patient data
    @param patient_data: pd.DataFrame: the patient's ECG, EDA and EMG  data.
    @param alpha: float: the regularization parameter
    @param max_iter: int: the maximum number of iterations
    @param dataset_folder: str: the folder to save the results to
    @param patient_id: int: the patient's id
    @param type: str: the type of data (ecg, eda or emg)
    :return: pd.DataFrame: the predicted stress levels
    """

    # Split data into X and y:
    x_train, x_test, y_train, y_test = split_patient_data(patient_data)

    # Scale the data:
    x_train, x_test = scale_data(x_train, x_test)

    # Fit the model:
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    lasso.fit(x_train, y_train)

    # Predict:
    y_pred = lasso.predict(x_test)

    # Create dataframe with predicted values:
    y_pred_df = pd.DataFrame(y_pred, columns=['Stress'])

    # Save the feature importance:
    feature_coefficients = pd.DataFrame(lasso.coef_, index=x_train.columns, columns=['Coefficient'])
    # Sort the faeture coefficients:
    feature_coefficients = feature_coefficients.sort_values(by='Coefficient', ascending=True)

    feature_coefficients.to_csv(RESULTS / dataset_folder / 'lasso_regression' / f'feature_coefficients_{patient_id}.csv')

    # Save the test scores:
    test_scores = pd.DataFrame(lasso.score(x_test, y_test), index=['R2'], columns=['Score'])
    test_scores.to_csv(RESULTS / dataset_folder / 'lasso_regression' / f'{type}_score_{patient_id}.csv')

    return y_pred_df


def scale_data(X_train:pd.DataFrame, X_test:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Auxiliary function to perform scaling for lasso
    :param X_train: pd.DataFrame: The training data
    :param X_test: pd.DataFrame: The testing data
    :return: the scaled data
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)


def main() -> None:
    """
    Perform lasso regression on each patient's data
    :return: None. Saves the results to a csv file.
    """

    for patient_id in range(1, 11):

        for task in ['cobot', 'manual']:

            # Load the data:
            ecg, eda, emg = load_patient_data(patient_id, task)

            # handle missing values
            ecg, eda, emg = handle_missing_values(ecg), handle_missing_values(eda), handle_missing_values(emg)

            lasso_regression(ecg, dataset_folder=f'{task}_results', patient_id=patient_id, type='ecg')
            lasso_regression(eda, dataset_folder=f'{task}_results', patient_id=patient_id, type='eda')
            lasso_regression(emg, dataset_folder=f'{task}_results', patient_id=patient_id, type='emg')

        # Concatenate the data:
        data = pd.concat([ecg, eda, emg], axis=1)

        # remove duplicate columns
        data = data.loc[:, ~data.columns.duplicated()]

        # Perform lasso regression:
        lasso_regression(data, dataset_folder='merged_results', patient_id=patient_id, type='all')


if __name__ == '__main__':
    main()
