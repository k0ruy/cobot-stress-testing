# extract stress labels from the questionnaires pickle file:
# Libraries:
import pickle
from pathlib import Path
import pandas as pd
import tabulate

from config import QUESTIONNAIRES


def main() -> None:
    """
    Extract the stress labels from the questionnaires pickle file.
    :return: None. Saves the stress labels in a csv file.
    """
    with open(QUESTIONNAIRES, "rb") as f:
        subject_data = pickle.load(f)

        # group by ID and get the mean of the stress column:
        stress = pd.DataFrame(subject_data.groupby('ID')['Stress'].mean())

        # group by ID and Task and get the mean of the stress column:
        stress_task = pd.DataFrame(subject_data.groupby(['ID', 'Task'])['Stress'].mean())
        stress_task.rename(columns={'Stress': 'avg_stress'}, inplace=True)

        # save to csv in the label folder:
        stress.to_csv(Path('..', 'label', 'stress_by_task.csv'))

        # rename the stress column to AvgStress:
        stress.rename(columns={'Stress': 'AvgStress'}, inplace=True)

        # save the stress labels to a csv file in the label folder:
        file_path = Path('..', 'label')
        file_path.mkdir(parents=True, exist_ok=True)
        stress.to_csv(Path('avg_stress.csv'))

        # get the baseline stress, denoted by Task "MANUAL":
        baseline_stress = subject_data[subject_data['Task'] == 'MANUAL'].groupby('ID')['Stress'].mean()

        # get the COBOT stress, denoted by Task "COBOT":
        cobot_stress = subject_data[subject_data['Task'] == 'COBOT'].groupby('ID')['Stress'].mean()

        # get the STROOP stress, denoted by Task "STROOP":
        stroop_stress = subject_data[subject_data['Task'] == 'STROOP'].groupby('ID')['Stress'].mean()

        # get the difference between the baseline and COBOT stress:
        stress_diff = cobot_stress - baseline_stress

        # get the difference between the baseline and STROOP stress:
        stroop_diff = stroop_stress - baseline_stress

        # get the difference between the COBOT and STROOP stress:
        cobot_stroop_diff = stroop_stress - cobot_stress

        # create a dataframe with the stress differences:
        stress_diff_df = pd.DataFrame({'ID': stress_diff.index,
                                       'StressDiff': stress_diff.values,
                                       'StroopDiff': stroop_diff.values,
                                       'CobotStroopDiff': cobot_stroop_diff.values})

        # save the stress differences to a csv file in the label folder:
        stress_diff_df.to_csv(Path('stress_diff.csv'), index=False)


# Driver code:
if __name__ == '__main__':
    main()



