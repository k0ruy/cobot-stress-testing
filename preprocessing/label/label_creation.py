# extract stress labels from the questionnaires pickle file:
# Libraries:
import pickle
from pathlib import Path
import pandas as pd
import tabulate

from config import QUESTIONNAIRES_CLEANED, SAVED_DATA


def main() -> None:
    """
    Extract the stress labels from the questionnaires pickle file.
    :return: None. Saves the stress labels in a csv file.
    """

    with open(QUESTIONNAIRES_CLEANED, "rb") as f:
        questionnaires = pickle.load(f)

        # for each patient-id folder in the saved data folder:
        for folder in SAVED_DATA.iterdir():
            if folder.is_dir():
                # get the patient id:
                patient_id = int(folder.name.split('-')[1])

                # get the questionnaires for the patient:
                q = questionnaires[questionnaires['ID'] == patient_id]

                # rename the number with file_number:
                q = q.rename(columns={'Number': 'File_number'})

                # get the stress labels:
                stress_labels = q[['Task', 'File_number', 'Stress']]

                # save the stress labels in a csv file:
                stress_labels.to_csv(folder / 'stress_labels.csv', index=False)


if __name__ == '__main__':
    main()










