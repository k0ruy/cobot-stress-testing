# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from pathlib import Path
# Modelling:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Global variables:
from config import SAVED_DATA


def load_data(path):
    """
    Load data from a csv file
    :param path: path to the csv file
    :return: pandas dataframe
    """
    return pd.read_csv(path)


def handle_missing_values(df):
    """
    Handle missing values
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    return df.dropna()


def split_data(data):

    """
    Split data into train and test sets
    :param data: pandas dataframe
    :return: train and test sets
    """
    X = data.drop(columns=["Stress"])
    y = data["Stress"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def main():

    X_train, X_test, y_train, y_test = split_data(
        handle_missing_values(
            load_data(Path(SAVED_DATA, "ecg_cobot.csv"))
        )
    )

    # random forest regressor:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # fit the model:
    rf.fit(X_train, y_train)

    # predict:
    y_pred = rf.predict(X_test)

    # round the values to the nearest integer, since the stress is expressed in integers:
    y_pred = np.round(y_pred)

    # evaluate:
    print("Random Forest Regressor")
    print("Train score: ", rf.score(X_train, y_train))
    print("Test score: ", rf.score(X_test, y_test))

    # Use other metrics to evaluate the model:
    print("Mean absolute error: ", mean_absolute_error(y_test, y_pred))
    print("Mean squared error: ", mean_squared_error(y_test, y_pred))
    print("Root mean squared error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 score: ", r2_score(y_test, y_pred))


# Driver:
if __name__ == '__main__':
    main()





