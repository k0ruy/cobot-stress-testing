# extract stress labels from the questionnaires pickle file:

# Libraries:
import pickle
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Driver:
if __name__ == '__main__':
    # Load the questionnaires pickle:
    with open(Path('..', '..', "questionnaires.pkl"), "rb") as f:
        subject_data = pickle.load(f)

        # group by ID and get the mean of the stress column:
        stress = pd.DataFrame(subject_data.groupby('ID')['Stress'].mean())

        # print the stress labels with tabulate:
        print(tabulate(stress, headers='keys', tablefmt='psql'))