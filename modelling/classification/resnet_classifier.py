# Libraries:
# Data manipulation:
from pathlib import Path

import numpy as np
import pandas as pd
# Modelling:
from keras.layers import Input, Conv1D, Add, Activation, MaxPooling1D, Flatten, Dense, BatchNormalization
from keras.models import Model
from keras.utils.np_utils import to_categorical
from modelling.classification.rf_stress_classification import load_data, handle_missing_values, split_data
from modelling.regression.nns_on_extracted_features import plot_history, scale_and_normalize_data
from sklearn.metrics import confusion_matrix
# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables:
from config import COBOT_RESULTS, MANUAL_RESULTS, SAVED_DATA, RESULTS


def resnet(input_shape, num_classes) -> Model:
    """
    Function to generate a resnet classifier for stress classification.
    @param input_shape: int: the input shape of the data
    @param num_classes: int: the number of classes, 3 in our case.
    :return: Model: The resnet model
    """
    # Input layer
    inputs = Input(shape=input_shape)

    # First convolutional layer with batch normalization and activation
    x = Conv1D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First residual block
    y = Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(64, kernel_size=3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([x, y])
    x = Activation('relu')(y)

    # Second residual block
    y = Conv1D(64, kernel_size=3, strides=1, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(64, kernel_size=3, strides=1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([x, y])
    x = Activation('relu')(y)

    # Max pooling layer
    x = MaxPooling1D(pool_size=2, strides=2)(x)

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs, x)

    return model


def main():
    """
    Main function to run the resnet classifier on the data
    :return: None. Saves the results in the results folder.
    """
    for task in ['cobot', 'manual']:
        # Define the path to use:
        path = Path(COBOT_RESULTS) if task == 'cobot' else Path(MANUAL_RESULTS)

        # Load the data:
        X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"ecg_{task}.csv")
            ), add_demographics=True, classify=True
        )

        X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"eda_{task}.csv")
            ), classify=True
        )

        X_train_emg, X_test_emg, y_train_emg, y_test_emg = split_data(
            handle_missing_values(
                load_data(SAVED_DATA / f"emg_{task}.csv")
            ), classify=True
        )

        # concatenate x and y train and test sets for total data prediction
        X_train = pd.concat([X_train_ecg, X_train_eda, X_train_emg], axis=1)
        X_test = pd.concat([X_test_ecg, X_test_eda, X_test_emg], axis=1)
        y_train = y_train_ecg  # all y_train and y_test are the same
        y_test = y_test_ecg

        # Scale and normalize the features:
        X_train, X_test = scale_and_normalize_data(X_train, X_test)

        # Reshape the data:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Convert 1D arrays to 2D one-hot encoded arrays for y_train and y_test
        y_train_categorical = to_categorical(y_train, 3)
        y_test_categorical = to_categorical(y_test, 3)

        # We try fixing the unbalanced dataset by using class weights:
        class_weights = {
            0: 1.,  # low
            1: 20.,  # medium
            2: 20.  # high
        }

        # Generate the model:
        model = resnet(X_train.shape[1:], 3)

        # Compile the model:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC', 'accuracy'])

        # Fit the model:
        history = model.fit(X_train, y_train_categorical, epochs=10, batch_size=64,
                            validation_data=(X_test, y_test_categorical), class_weight=class_weights)

        # Path to save the results to:
        path_to_save = RESULTS / f"{task}_results" / "resnet"
        path_to_save.mkdir(parents=True, exist_ok=True)

        # Plot the history:
        plot_history(history, path_to_save, 'resnet')

        # Confusion matrix:
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = y_test_categorical.argmax(axis=1)

        cm = confusion_matrix(y_test, y_pred)
        cm = pd.DataFrame(cm, columns=['low', 'medium', 'high'], index=['low', 'medium', 'high'])
        cm.to_csv(path_to_save / 'confusion_matrix.csv')

        # plot confusion matrix:
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix of ResNet Stress Classification')
        plt.savefig(path_to_save / f'confusion_matrix_{task}.png')


if __name__ == '__main__':
    main()
