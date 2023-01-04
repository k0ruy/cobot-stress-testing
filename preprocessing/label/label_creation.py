# extract stress labels from the questionnaires pickle file:
# Libraries:
# Data manipulation:
import pickle
from pathlib import Path
import pandas as pd

# Global variables:
from config import QUESTIONNAIRES_CLEANED, SAVED_DATA, EXPERIMENTS


def match_filename_with_file_number():
    """
    Match the file name with the file number.
    :return: None, adds a column to the questionnaires csv file.
    """
    for task in ['cobot', 'manual']:
        for folder in Path(EXPERIMENTS, task).iterdir():
            if folder.is_dir():
                # get the patient id:
                patient_id = folder.name.split('-')[1]

                # get all the txt files in the folder preserving the order:
                files = sorted(folder.glob('*.txt'))
                # map the file number to the file name:
                file_number_to_file_name = {i: f.name for i, f in enumerate(files, start=1)}

                # get the questionnaires for the patient in the SAVE_DATA folder:
                q = pd.read_csv(SAVED_DATA / f'patient-{patient_id}' / f'questionnaire_{task}.csv')

                # add the file name column:
                q['filename'] = q['File_number'].map(file_number_to_file_name)

                # save the questionnaires in the SAVE_DATA folder:
                q.to_csv(SAVED_DATA / f'patient-{patient_id}' / f'questionnaire_{task}.csv', index=False)


def merge_features_with_questionnaires():
    """
    Merge the features with the questionnaires.
    :return: None, saves the merged features in a csv file.
    """

    # for each patient-id folder in the saved data folder:
    for folder in SAVED_DATA.iterdir():
        if folder.is_dir():
            # get the patient id:
            patient_id = int(folder.name.split('-')[1])

            for task in ['cobot', 'manual']:
                # get the ECG features:
                ecg_features = pd.read_csv(folder / f'ecg_{task}_{patient_id}.csv')
                # get the EDA features:
                eda_features = pd.read_csv(folder / f'eda_{task}_{patient_id}.csv')
                # get the EMG features:
                emg_features = pd.read_csv(folder / f'emg_{task}_{patient_id}.csv')

                # get the questionnaires:
                questionnaires = pd.read_csv(folder / f'questionnaire_{task}.csv')

                # merge each feature with the questionnaires on the file name:
                ecg_features = ecg_features.merge(questionnaires, on='filename')
                eda_features = eda_features.merge(questionnaires, on='filename')
                emg_features = emg_features.merge(questionnaires, on='filename')

                # save the merged features in a csv file:
                ecg_features.to_csv(folder / f'ecg_{task}_{patient_id}.csv', index=False)
                eda_features.to_csv(folder / f'eda_{task}_{patient_id}.csv', index=False)
                emg_features.to_csv(folder / f'emg_{task}_{patient_id}.csv', index=False)


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

                for task in ['cobot', 'manual', 'rest', 'stroopeasy', 'stroophard']:

                    # get the questionnaires for the patient:
                    q = questionnaires[questionnaires['ID'] == patient_id]
                    q = q[q['Task'] == task.upper()]

                    # if the questionnaires are empty, skip:
                    # we provide this for future use, in case we want to add more questionnaires for rest and stroop.
                    if q.empty:
                        continue

                    # rename the column Number to File_number:
                    q = q.rename(columns={'Number': 'File_number'})

                    # save the stress labels in a csv file:
                    q.to_csv(folder / f'questionnaire_{task}.csv', index=False)

        # match the file name with the file number:
        match_filename_with_file_number()

        # merge the features with the questionnaires:
        merge_features_with_questionnaires()


if __name__ == '__main__':
    main()










