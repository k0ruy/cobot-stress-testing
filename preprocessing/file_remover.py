# remove the row with file  06-05-30_10-53-13_manual from the pickle file

# Libraries:
import pickle
from pathlib import Path
import pandas as pd


# Driver:

if __name__ == '__main__':
    # Load the questionnaires pickle:
    with open(Path('..', "questionnaires.pkl"), "rb") as f:
        subject_data = pickle.load(f)

        # remove the row with index 73 as it is the row with the file 06-05-30_10-53-13_manual:
        subject_data = subject_data.drop(subject_data.iloc[73].name)

        # duplicate all rows with user ID 10:
        subject_11 = subject_data[subject_data['ID'] == 10].copy()

        # change the ID to 11:
        subject_11['ID'] = 11

        # concatenate the two dataframes:
        subject_data = pd.concat([subject_data, subject_11], axis=0)

    with open(Path('..', 'questionnaires.pkl'), 'wb') as f:
        pickle.dump(subject_data, f)


