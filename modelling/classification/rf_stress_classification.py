# Library to classify stress for the cobot and manual experiments
# in low, medium and high values.
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Modelling:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from modelling.regression.rf_stress_prediction import load_data, handle_missing_values, split_data

# Global variables
from config import SAVED_DATA, COBOT_RESULTS, MANUAL_RESULTS, PLOTS

def main():
    """
    Main function to run the random forest model on the data
    :return: None. Saves the results and plots to the results folder
    """

    for task in ['cobot', 'manual']:

        # load data:
        X_train_ecg_demo, X_test_ecg_demo, y_train_ecg_demo, y_test_ecg_demo = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'ecg_{task}.csv')), add_demographics=True, classify=True)

        X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'ecg_{task}.csv')), add_demographics=False, classify=True)

        X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'eda_{task}.csv')), add_demographics=False, classify=True)

        X_train_emg, X_test_emg, y_train_emg, y_test_emg = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'emg_{task}.csv')), add_demographics=False, classify=True)

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
        rf_ecg = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_eda = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_emg = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_all_signals = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_ecg_eda = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_added_demographics = RandomForestClassifier(n_estimators=100, random_state=42)

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
        with open(path / f"rf_{task}_classification_evaluation.txt", "w") as f:
            f.write("ECG stress prediction results with an un-tuned random forest classifier:")
            f.write("\n")
            f.write(f"Train score accuracy: {rf_ecg.score(X_train_ecg, y_train_ecg):.3f}")
            f.write("\n")
            f.write(f"Test score accuracy: {rf_ecg.score(X_test_ecg, y_test_ecg):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test_ecg, y_pred_ecg, average='weighted'):.3f}")
            f.write("\n")
            f.write("EDA stress prediction results with an un-tuned random forest classifier:")
            f.write("\n")
            f.write(f"Train score: {rf_eda.score(X_train_eda, y_train_eda):.3f}")
            f.write("\n")
            f.write(f"Test score: {rf_eda.score(X_test_eda, y_test_eda):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test_eda, y_pred_eda, average='weighted'):.3f}")
            f.write("\n")
            f.write("EMG stress prediction results with an un-tuned random forest classifier:")
            f.write("\n")
            f.write(f"Train score accuracy: {rf_emg.score(X_train_emg, y_train_emg):.3f}")
            f.write("\n")
            f.write(f"Test score accuracy: {rf_emg.score(X_test_emg, y_test_emg):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test_emg, y_pred_emg, average='weighted'):.3f}")
            f.write("\n")
            f.write("All stress prediction results with an un-tuned random forest classifier:")
            f.write("\n")
            f.write(f"Train score accuracy: {rf_all_signals.score(X_train, y_train):.3f}")
            f.write("\n")
            f.write(f"Test score accuracy: {rf_all_signals.score(X_test, y_test):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
            f.write("\n")
            f.write("ECG and EDA stress prediction results with an un-tuned random forest classifier:")
            f.write("\n")
            f.write(f"Train score accuracy: {rf_ecg_eda.score(X_train_ecg_eda, y_train_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"Test score accuracy: {rf_ecg_eda.score(X_test_ecg_eda, y_test_ecg_eda):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test_ecg_eda, y_pred_ecg_eda, average='weighted'):.3f}")
            f.write("\n")
            f.write(f"Train score accuracy: "
                    f"{rf_added_demographics.score(X_train_ecg_eda_demo, y_train_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"Test score accuracy: "
                    f"{rf_added_demographics.score(X_test_ecg_eda_demo, y_test_ecg_eda_demo):.3f}")
            f.write("\n")
            f.write(f"F1 score: {f1_score(y_test_ecg_eda_demo, y_pred_ecg_eda_demo, average='weighted'):.3f}")
            f.write("\n")

        # Plot the feature importance
        feature_importance = rf_all_signals.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        # get the top 10 features
        top_10 = sorted_idx[-10:]
        pos = np.arange(top_10.shape[0]) + .5
        plt.figure(figsize=(12, 6))
        plt.barh(pos, feature_importance[top_10], align='center')
        plt.yticks(pos, X_train.columns[top_10])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        Path(PLOTS).mkdir(parents=True, exist_ok=True)
        plt.savefig(PLOTS / f"rf_{task}_classification_feature_importance.png")


if __name__ == '__main__':
    main()
