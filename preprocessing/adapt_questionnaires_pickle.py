# remove the row with file  06-05-30_10-53-13_manual from the pickle file

# Libraries:
import pickle
import pandas as pd
from config import QUESTIONNAIRES, SAVED_DATA

# remove the row limit in the console:
pd.set_option('display.max_rows', None)


def main() -> None:
    """
    Remove the row from file  06-05-30_10-53-13_manual from the pickle file.
    Corrects the data based on slide 23 of professor Baraldo's project presentation.
    :return: None. Saves the cleaned pickle file.
    """

    # Load the questionnaires pickle:
    with open(QUESTIONNAIRES, "rb") as f:
        questionnaires = pickle.load(f)

        # Visual inspection of the data:
        print(questionnaires.head(2000))

        q_c = questionnaires.copy()

        # remove the row with index 73 as it is the row with the file 06-05-30_10-53-13_manual:
        q_c = q_c.drop(q_c.iloc[73].name)

        # TODO: Chri, is this needed?
        # duplicate all rows with user ID 10:
        # q_c_11 = q_c[q_c['ID'] == 10].copy()

        # change the ID to 11:
        # q_c_11['ID'] = 11

        # concatenate the two dataframes:
        # q_c = pd.concat([q_c, q_c_11], axis=0)

        # Drop number == 3 where ID == 1 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 1) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 3)].index)
        # Drop Number == 5 where ID == 1 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 1) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 5)].index)
        # Shift the numbers of the files with ID == 1 and Task == MANUAL appropriately:
        q_c.loc[(q_c['ID'] == 1) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 4), 'Number'] = 3
        for i in range(6, 13):
            q_c.loc[(q_c['ID'] == 1) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == i), 'Number'] = i - 2

        # Drop number == 7 where ID == 2 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 2) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 7)].index)
        # Shift the numbers of the files with ID == 2 and Task == MANUAL appropriately:
        for i in range(8, 16):
            q_c.loc[(q_c['ID'] == 2) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == i), 'Number'] = i - 1

        # Drop number == 3 where ID == 3 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 3) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 3)].index)
        # Shift the numbers of the files with ID == 3 and Task == MANUAL appropriately:
        for i in range(4, 16):
            q_c.loc[(q_c['ID'] == 3) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == i), 'Number'] = i - 1

        # Drop number == 11 where ID == 3 the task is COBOT:
        q_c = q_c.drop(q_c[(q_c['ID'] == 3) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == 11)].index)
        # Shift the numbers of the files with ID == 3 and Task == COBOT appropriately:
        for i in range(12, 16):
            q_c.loc[(q_c['ID'] == 3) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == i), 'Number'] = i - 1

        # Drop number == 0 where ID == 7 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 7) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 0)].index)
        # TODO: this does not exist in the original questionnaire, bug in the slides or we should consider 0 as 1?
        # Shift the numbers of the files with ID == 7 and Task == MANUAL appropriately:
        # for i in range(1, 16):
            # q_c.loc[(q_c['ID'] == 7) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == i), 'Number'] = i - 1

        # Drop number == 9 where ID == 7 the task is COBOT:
        q_c = q_c.drop(q_c[(q_c['ID'] == 7) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == 9)].index)
        # Shift the numbers of the files with ID == 7 and Task == COBOT appropriately:
        for i in range(10, 16):
            q_c.loc[(q_c['ID'] == 7) & (q_c['Task'] == 'COBOT') & (q_c['Number'] == i), 'Number'] = i - 1

        # Drop number == 5 where ID == 9 the task is MANUAL:
        q_c = q_c.drop(q_c[(q_c['ID'] == 9) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == 5)].index)
        # Shift the numbers of the files with ID == 9 and Task == MANUAL appropriately:
        for i in range(6, 16):
            q_c.loc[(q_c['ID'] == 9) & (q_c['Task'] == 'MANUAL') & (q_c['Number'] == i), 'Number'] = i - 1

        # Shift the numbers of the files with ID == 10 and Task == STROOP appropriately:
        for i in range(3, 5):
            q_c.loc[(q_c['ID'] == 10) & (q_c['Task'] == 'STROOP') & (q_c['Number'] == i), 'Number'] = i - 1

        # save the cleaned questionnaires:
        q_c.to_pickle(SAVED_DATA / "questionnaires_cleaned.pkl")

        # save the csv:
        q_c.to_csv(SAVED_DATA / "questionnaires_cleaned.csv", index=False)

        # sort by ID, Task and Number:
        q_c = q_c.sort_values(by=['ID', 'Task', 'Number'])

        # Visual inspection of the cleaned questionnaires:
        print(q_c.head(2000))


# Driver:
if __name__ == '__main__':
    main()
