# tuner for the random forest model on the merged dataset:
from pathlib import Path

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV  # noqa
from sklearn.metrics import r2_score

from config import SAVED_DATA, COBOT_RESULTS
from rf_stress_prediction import load_data, handle_missing_values, split_data


def main():
    path = Path(COBOT_RESULTS)

    # Load the data:
    X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(
        handle_missing_values(
            load_data(SAVED_DATA / f"ecg_cobot.csv")
        ), add_demographics=True
    )

    X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_data(
        handle_missing_values(
            load_data(SAVED_DATA / f"eda_cobot.csv")
        )
    )

    X_train_emg, X_test_emg, y_train_emg, y_test_emg = split_data(
        handle_missing_values(
            load_data(SAVED_DATA / f"emg_cobot.csv")
        )
    )

    # concatenate x and y train and test sets for total data prediction
    X_train = pd.concat([X_train_ecg, X_train_eda, X_train_emg], axis=1)
    X_test = pd.concat([X_test_ecg, X_test_eda, X_test_emg], axis=1)
    y_train = y_train_ecg  # all y_train and y_test are the same
    y_test = y_test_ecg

    # define the parameters to tune
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 300, 400, 500],
        'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'min_samples_split': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    }
    # create a base model
    rf = RandomForestRegressor()
    # define the grid search
    s_search = HalvingRandomSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_jobs=-1,
        cv=2,
        verbose=2,
    )
    # fit the grid search
    s_search.fit(X_train, y_train)
    # print the best parameters
    print(s_search.best_params_)
    # predict the test set
    y_pred = np.round(s_search.predict(X_test))
    # print the r^2 score
    print(r2_score(y_test, y_pred))


if __name__ == '__main__':
    main()
