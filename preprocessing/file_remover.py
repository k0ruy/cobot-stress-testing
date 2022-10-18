# remove the row with file  06-05-30_10-53-13_manual from the pickle file

# Libraries:
import pickle
from pathlib import Path


# Driver:

if __name__ == '__main__':
    with open(Path('..', 'questionnaires.pkl'), 'rb') as f:
        subject_data = pickle.load(f)

        # drop row 73:
        subject_data.drop(73, inplace=True)

        # save the new pickle file:
        pickle.dump(subject_data, open(Path('..', 'questionnaires.pkl'), 'wb'))

