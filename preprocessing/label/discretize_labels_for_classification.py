# Library to discretize stress labels for classification:

# Data manipulation:
import pandas as pd
# Modelling:
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Global variables:
from config import SAVED_DATA


# Functions:
def discretize_labels_for_classification(labels: pd.Series, n_bins: int = 3) -> pd.Series:
    """
    Discretize the stress labels for classification
    @param labels: The stress labels
    @param n_bins: The number of bins to use
    @return: The discrete labels, 0 is not stressed, 1 is stressed, 2 is very stressed
    """
    # Discretize the labels:
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    # Split the labels into y_train and y_test with the usual 80/20 split and random seed 42:
    y_train, y_test = train_test_split(labels, test_size=0.2, random_state=42)
    # Fit the discretizer on the training labels:
    discretizer.fit(y_train.values.reshape(-1, 1))
    # Transform the training and testing labels:
    y_train = discretizer.transform(y_train.values.reshape(-1, 1))
    y_test = discretizer.transform(y_test.values.reshape(-1, 1))
    # Return the discrete labels concatenated:
    return pd.concat([pd.Series(y_train.flatten()), pd.Series(y_test.flatten())], ignore_index=True)


def main() -> None:
    """
    Main function to add a discrete stress label column to the data
    """
    # Load the data:
    for task in ['cobot', 'manual']:
        # read the 3 csv files:
        ecg = pd.read_csv(SAVED_DATA / f"ecg_{task}.csv")
        eda = pd.read_csv(SAVED_DATA / f"eda_{task}.csv")
        emg = pd.read_csv(SAVED_DATA / f"emg_{task}.csv")

        # discretize the stress labels, we could also reuse the same discretizer for all 3 csv files since
        # they all have the same stress labels:
        ecg['discrete_stress'] = discretize_labels_for_classification(ecg['Stress'])
        eda['discrete_stress'] = discretize_labels_for_classification(eda['Stress'])
        emg['discrete_stress'] = discretize_labels_for_classification(emg['Stress'])

        # save the data in the original
        ecg.to_csv(SAVED_DATA / f"ecg_{task}.csv", index=False)
        eda.to_csv(SAVED_DATA / f"eda_{task}.csv", index=False)
        emg.to_csv(SAVED_DATA / f"emg_{task}.csv", index=False)


if __name__ == '__main__':
    main()
