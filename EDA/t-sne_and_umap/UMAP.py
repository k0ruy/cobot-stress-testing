# Libraries:
# Data manipulation:
import pandas as pd
# Dimensionality reduction:
import umap
from modelling.rf_stress_prediction import load_data, handle_missing_values, split_data
from t_SNE import merge_data

# Global variables:
from config import SAVED_DATA


def main() -> None:
    """
    Performs T-sne on the COBOT and MANUAL experiments
    :return: None. Saves the T-sne data to a csv file.
    """
    # Load data
    ecg_c: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'ecg_cobot.csv'))
    ecg_m: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'ecg_manual.csv'))
    eda_c: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'eda_cobot.csv'))
    eda_m: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'eda_manual.csv'))
    emg_c: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'emg_cobot.csv'))
    emg_m: pd.DataFrame = handle_missing_values(load_data(SAVED_DATA / 'emg_manual.csv'))

    # merge all c and m files by column:
    cobot = merge_data(dataframes=[ecg_c, eda_c, emg_c], axis=1)
    manual = merge_data(dataframes=[ecg_m, eda_m, emg_m], axis=1)

    # merge cobot and manual by rows:
    cobot_and_manual = merge_data(dataframes=[cobot, manual], axis=0)

    # split the dataset:
    X_train, X_test, y_train, y_test = split_data(cobot_and_manual, add_demographics=True, classify=False)

    # merge the features since we are inspecting the data:
    X = merge_data(dataframes=[X_train, X_test], axis=0)

    # save the stress target:
    y = merge_data(dataframes=[y_train, y_test], axis=0)

    # if we are using discrete stress, rename it to Stress to have a single target name for EDA visualizations:
    if "discrete_stress" in y.columns:
        y.rename(columns={"discrete_stress": "Stress"}, inplace=True)

    y.to_csv(SAVED_DATA / "stress_for_umap.csv")

    # fit
    umap_results = umap.UMAP(n_neighbors=50, n_components=3).fit_transform(X, y=y)

    # save the results as a dataframe, add the CustomerID as the first column:
    umap_results_df = pd.DataFrame(umap_results)

    umap_results_df.to_csv(SAVED_DATA / f'cobot_manual_umap.csv', index=False)


# Driver:
if __name__ == '__main__':
    main()

