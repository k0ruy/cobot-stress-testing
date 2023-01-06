# Import necessary libraries
import pickle
from pathlib import Path
import keras.callbacks
import numpy as np
import pandas as pd
from keras.losses import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

from simple_stress_predictor import load_data, handle_missing_values, split_data
import matplotlib.pyplot as plt

# Global variables:
from config import SAVED_DATA, COBOT_RESULTS, MANUAL_RESULTS

# Tensorflow logging 3:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def plot_history(history: keras.callbacks.History, path: Path, model_type: str = "ff_nn",
                 title: str = "Feed forward neutral network history") -> None:
    """
    Plot the history of the model
    """
    # Plot the loss:
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.savefig(Path(path, f'{model_type}_ecg_model_history.png'))
    plt.show()


def nn_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
             path: Path) -> keras.Model:
    """
    Generate the feed forward neural network model
    @param X_train: The training data
    @param X_test: The testing data
    @param y_train: The training stress labels
    @param y_test: The testing stress labels
    @param path: Path: The path to save the plots to.
    :return: ff_nn_model: keras.Model: The feed forward neural network model
    """
    # Create the model for the feed forward neural network:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Plot the history:
    plot_history(history, path)
    return model


def lstm_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
               path: Path) -> keras.Model:
    """
    Generate the LSTM model
    @param X_train: The training data
    @param X_test: The testing data
    @param y_train: The training stress labels
    @param y_test: The testing stress labels
    @param path: Path: The path to save the plots to.
    :return:  lstm model: keras.Model: The LSTM model
    """

    # create the model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
        LSTM(32, return_sequences=False),
        Dense(1)
    ])

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Plot the history:
    plot_history(history, model_type="lstm", path=path, title="LSTM model history")

    return model


def scale_and_normalize_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def main():

    for task in ['cobot', 'manual']:

        # Define the path to use:
        path = Path(COBOT_RESULTS) if task == 'cobot' else Path(MANUAL_RESULTS)

        # Load the data:
        X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"ecg_{task}.csv")
            ), add_demographics=True
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

        # concatenate x and y train and test sets for total data prediction
        X_train = pd.concat([X_train_ecg, X_train_eda, X_train_emg], axis=1)
        X_test = pd.concat([X_test_ecg, X_test_eda, X_test_emg], axis=1)
        y_train = y_train_ecg  # all y_train and y_test are the same
        y_test = y_test_ecg

        # Scale and normalize the features:
        X_train, X_test = scale_and_normalize_data(X_train, X_test)

        # Generate the models:
        ff_nn_model = nn_model(X_train, X_test, y_train, y_test, path)
        lstm_nn_model = lstm_model(X_train, X_test, y_train, y_test, path)

        # predict the stress labels, rounded to the nearest integer:
        y_pred_ff_nn = np.round(ff_nn_model.predict(X_test))
        y_pred_lstm = np.round(lstm_nn_model.predict(X_test))

        # calculate the mean squared error on the training data:
        mse_ff_nn_train = mean_squared_error(y_train, np.round(ff_nn_model.predict(X_train)))
        mse_lstm_train = mean_squared_error(y_train, np.round(lstm_nn_model.predict(X_train)))

        # calculate the mean squared error:
        mse_ff_nn = mean_squared_error(y_test, y_pred_ff_nn)
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)

        # calculate the r2 score:
        r2_ff_nn = r2_score(y_test, y_pred_ff_nn)
        r2_lstm = r2_score(y_test, y_pred_lstm)

        with open(path / f"NN_{task}_evaluation.txt", "w") as f:
            f.write("ECG stress prediction results with an un-tuned random forest regressor:")
            f.write("\n")
            f.write(f"Train score feed forward: {mse_ff_nn_train:.3f}")
            f.write("\n")
            f.write(f"Train score LSTM: {mse_lstm_train:.3f}")
            f.write("\n")
            f.write(f"Test score feed forward: {mse_ff_nn:.3f}")
            f.write("\n")
            f.write(f"Test score LSTM: {mse_lstm:.3f}")
            f.write("\n")
            f.write(f"R2 score feed forward: {r2_ff_nn:.3f}")
            f.write("\n")
            f.write(f"R2 score LSTM: {r2_lstm:.3f}")
            f.write("\n")


if __name__ == '__main__':
    main()

