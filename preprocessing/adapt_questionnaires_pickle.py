# remove the row with file  06-05-30_10-53-13_manual from the pickle file

# Libraries:
import pickle
from pathlib import Path
import pandas as pd

from config import QUESTIONNAIRES, SAVED_DATA

# Driver:

if __name__ == '__main__':
    # Load the questionnaires pickle:
    with open(QUESTIONNAIRES, "rb") as f:
        questionnaires = pickle.load(f)
        q_c = questionnaires.copy()

        # remove the row with index 73 as it is the row with the file 06-05-30_10-53-13_manual:
        q_c = q_c.drop(q_c.iloc[73].name)

        # duplicate all rows with user ID 10:
        q_c_11 = q_c[q_c['ID'] == 10].copy()

        # change the ID to 11:
        q_c_11['ID'] = 11

        # concatenate the two dataframes:
        q_c = pd.concat([q_c, q_c_11], axis=0)

        # Drop number == 3 and Number == 5 where ID == 1 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 1) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 3) |
                           (q_c['Number'] == 5)].index)

        # Drop number == 7 where ID == 2 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 2) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 7)].index)

        # Drop number == 3 where ID == 3 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 3) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 3)].index)
        # Drop number == 11 where ID == 3 the task is COBOT:
        q_c = q_c.drop(q_c[(q_c['ID'] == 3) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == 11)].index)

        # Drop number == 0 where ID == 7 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 7) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 0)].index)
        # Drop number == 9 where ID == 7 the task is COBOT:
        q_c = q_c.drop(q_c[(q_c['ID'] == 7) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == 9)].index)

        # Drop number == 5 where ID == 9 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 9) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 5)].index)

        # save the cleaned questionnaires:
        q_c.to_pickle(SAVED_DATA / "questionnaires_cleaned.pkl")

