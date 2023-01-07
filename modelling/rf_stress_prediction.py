# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from pathlib import Path
# Modelling:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
# Plotting:
import matplotlib.pyplot as plt

# Global variables:
from config import SAVED_DATA, PLOTS, COBOT_RESULTS, MANUAL_RESULTS


def load_data(path: Path) -> pd.DataFrame:
    """
    Load data from a csv file
    :param path: path to the csv file
    :return: pandas dataframe
    """
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    df.dropna(axis=1, how='all', inplace=True)  # drop columns that have all nans

    assert df.isnull().sum().sum() == 0, "There are still missing values in the dataframe"

    return df.dropna()  # drop rows that have stray nans


def split_data(data, add_demographics=False, classify=False) -> \
        (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

    """
    Split data into train and test sets
    @param data: pandas dataframe
    @param add_demographics: boolean: whether to add demographics to the model's features.
    @param classify: boolean: whether to use classification or regression
    :return: train and test sets
    """
    columns_to_drop = ['patient', 'filename', 'Stress', 'discrete_stress', 'NARS_S3', 'NARS_S2', 'NARS_S1',
                       'PSS', 'NASA_Total', 'Frustration', 'Physical_Demand', 'Mental_Demand', 'Fatigue', 'STAI_Total',
                       'File_number', 'Task', 'ID'] if add_demographics else \
        ['Age', 'Experience', 'Sex', 'patient', 'filename', 'Stress', 'discrete_stress', 'NARS_S3', 'NARS_S2', 'NARS_S1', 'PSS',
         'NASA_Total', 'Frustration', 'Physical_Demand', 'Mental_Demand', 'Fatigue', 'STAI_Total', 'File_number',
         'Task', 'ID']

    # features:
    X = data.drop(columns=columns_to_drop)

    # labels:
    y = data["discrete_stress"] if classify else data["Stress"]

    # split the data into train and test sets using a 80/20 split and a random state of 42 for all models:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to run the random forest model on the data
    :return: None. Saves the results and plots to the results folder
    """

    for task in ['cobot', 'manual']:

        X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"ecg_{task}.csv")
            )
        )

        X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"eda_{task}.csv")
            )
        )

        X_train_emg, X_test_emg, y_train_emg, y_test_emg = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"emg_{task}.csv")
            )
        )

        X_train_ecg_demo, X_test_ecg_demo, y_train_ecg_demo, y_test_ecg_demo = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"ecg_{task}.csv")
            ), add_demographics=True
        )

        # concatenate x and y train and test sets for total data prediction
        X_train = pd.concat([X_train_ecg, X_train_eda, X_train_emg], axis=1)
        X_test = pd.concat([X_test_ecg, X_test_eda, X_test_emg], axis=1)
        y_train = y_train_ecg  # all y_train sets are the same
        y_test = y_test_ecg

        # concatenate only ECG and EDA data for ECG and EDA prediction
        X_train_ecg_eda = pd.concat([X_train_ecg, X_train_eda], axis=1)
        X_test_ecg_eda = pd.concat([X_test_ecg, X_test_eda], axis=1)
        y_train_ecg_eda = y_train_ecg  # all y_train sets are the same
        y_test_ecg_eda = y_test_ecg

        # concatenate demographics with ECG and EDA data for ECG and EDA prediction
        X_train_ecg_eda_demo = pd.concat([X_train_ecg_demo, X_train_eda], axis=1)
        X_test_ecg_eda_demo = pd.concat([X_test_ecg_demo, X_test_eda], axis=1)
        y_train_ecg_eda_demo = y_train_ecg  # all y_train sets are the same
        y_test_ecg_eda_demo = y_test_ecg

        # random forest regressor:
        rf_ecg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_eda = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_emg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_all_signals = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_ecg_eda = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_added_demographics = RandomForestRegressor(n_estimators=100, random_state=42)

        # fit the models:
        rf_ecg.fit(X_train_ecg, y_train_ecg)
        rf_eda.fit(X_train_eda, y_train_eda)
        rf_emg.fit(X_train_emg, y_train_emg)
        rf_all_signals.fit(X_train, y_train)
        rf_ecg_eda.fit(X_train_ecg_eda, y_train_ecg_eda)
        rf_added_demographics.fit(X_train_ecg_eda_demo, y_train_ecg_eda_demo)

        # predict, round the values to the nearest integer, since the stress is expressed in integers:
        y_pred_ecg = np.round(rf_ecg.predict(X_test_ecg))
        y_pred_eda = np.round(rf_eda.predict(X_test_eda))
        y_pred_emg = np.round(rf_emg.predict(X_test_emg))
        y_pred = np.round(rf_all_signals.predict(X_test))
        y_pred_ecg_eda = np.round(rf_ecg_eda.predict(X_test_ecg_eda))
        y_pred_ecg_eda_demo = np.round(rf_added_demographics.predict(X_test_ecg_eda_demo))

        # evaluate:
        if task == 'cobot':
            path = COBOT_RESULTS
        else:
            path = MANUAL_RESULTS

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(path / f"{task}_evaluation.txt", "w") as f:
            f.write("ECG stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_ecg.score(X_train_ecg, y_train_ecg):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_ecg.score(X_test_ecg, y_test_ecg):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test_ecg, y_pred_ecg):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test_ecg, y_pred_ecg):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test_ecg, y_pred_ecg)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test_ecg, y_pred_ecg):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test_ecg, y_pred_ecg):.3f}")
            f.write("\n")
            f.write("EDA stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_eda.score(X_train_eda, y_train_eda):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_eda.score(X_test_eda, y_test_eda):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test_eda, y_pred_eda):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test_eda, y_pred_eda):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test_eda, y_pred_eda)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test_eda, y_pred_eda):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test_eda, y_pred_eda):.3f}")
            f.write("\n")
            f.write("EMG stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_emg.score(X_train_emg, y_train_emg):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_emg.score(X_test_emg, y_test_emg):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test_emg, y_pred_emg):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test_emg, y_pred_emg):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test_emg, y_pred_emg)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test_emg, y_pred_emg):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test_emg, y_pred_emg):.3f}")
            f.write("\n")
            f.write("All stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_all_signals.score(X_train, y_train):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_all_signals.score(X_test, y_test):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test, y_pred):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test, y_pred):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test, y_pred):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test, y_pred):.3f}")
            f.write("\n")
            f.write("ECG and EDA stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_ecg_eda.score(X_train_ecg_eda, y_train_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_ecg_eda.score(X_test_ecg_eda, y_test_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test_ecg_eda, y_pred_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test_ecg_eda, y_pred_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test_ecg_eda, y_pred_ecg_eda)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test_ecg_eda, y_pred_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test_ecg_eda, y_pred_ecg_eda):.3f}")
            f.write("\n")
            f.write("ECG, EDA, and demographics stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score: {rf_added_demographics.score(X_train_ecg_eda_demo, y_train_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_added_demographics.score(X_test_ecg_eda_demo, y_test_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Mean absolute error: {mean_absolute_error(y_test_ecg_eda_demo, y_pred_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Mean squared error: {mean_squared_error(y_test_ecg_eda_demo, y_pred_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Root mean squared error: {np.sqrt(mean_squared_error(y_test_ecg_eda_demo, y_pred_ecg_eda_demo)):.3f}")
            f.write("\n")
            f.write(f"R2 score: {r2_score(y_test_ecg_eda_demo, y_pred_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Explained variance score: {explained_variance_score(y_test_ecg_eda_demo, y_pred_ecg_eda_demo):.3f}")
            f.write("\n")

        # plots:
        Path(PLOTS).mkdir(parents=True, exist_ok=True)

        # plot the feature importance:
        plt.figure(figsize=(10, 5))
        # select the top 10 features:
        feature_importance = pd.Series(rf_added_demographics.feature_importances_,
                                       index=X_train_ecg_eda_demo.columns).sort_values(ascending=False)[:10]
        feature_importance.plot(kind='bar')
        # add space for the x-axis labels:
        plt.subplots_adjust(bottom=0.3)
        # rotate the x-axis labels:
        plt.xticks(rotation=15)
        plt.title("Feature importance")
        plt.savefig(Path(PLOTS, f"{task}_feature_importance.png"))
        plt.show()


if __name__ == '__main__':
    main()





