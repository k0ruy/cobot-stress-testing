# Auxiliary library to visualize the stress and discrete stress levels

# Libraries:
# Data Manipulation:
from pathlib import Path
import pandas as pd


# Visualization:
import matplotlib as mpl
import matplotlib.pyplot as plt
from config import SAVED_DATA, PLOTS
from modelling.regression.rf_stress_prediction import load_data, split_data, handle_missing_values

mpl.use('tkagg')


def main() -> None:
    """
    Main function to run the script
    :return: None. Saves the plots to the plots folder
    """
    _, _, y_train_ecg, y_test_ecg = split_data(handle_missing_values(
        load_data(SAVED_DATA / f'ecg_cobot.csv')), add_demographics=False)

    y_stress = pd.concat([y_train_ecg, y_test_ecg])

    _, _, y_train_manual, y_test_manual = split_data(handle_missing_values(
        load_data(SAVED_DATA / f'ecg_cobot.csv')), add_demographics=False, classify=True)

    y_discrete_stress = pd.concat([y_train_manual, y_test_manual])


    # create a figure:
    fig = plt.figure(figsize=(16, 10))
    # create histogram:
    ax = fig.add_subplot(111)
    ax.hist(y_stress, bins=10, alpha=0.8, edgecolor='k')
    plt.title('Stress levels distribution')

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'Stress', f'stress_distribution.png'))
    except FileNotFoundError:
        Path(PLOTS / 'Stress').mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PLOTS / 'Stress', f'stress_distribution.png'))

    # create a figure:
    fig = plt.figure(figsize=(16, 10))
    # create a bar plot with the 3 stress levels:
    ax = fig.add_subplot(111)
    ax.bar(['Low', 'Medium', 'High'], y_discrete_stress.value_counts(), alpha=0.8, edgecolor='k')
    # add the values to the bars:
    for i, v in enumerate(y_discrete_stress.value_counts()):
        # center the text in the bar, add separation between the text and the bar:
        ax.text(i, v + 0.1, str(v), color='black', fontsize=14, ha='center')
    # hide the y axis ticks:
    ax.get_yaxis().set_ticks([])
    plt.title('Stress levels distribution')

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'Stress', f'discrete_stress_distribution.png'))
    except FileNotFoundError:
        Path(PLOTS / 'Stress').mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PLOTS / 'Stress', f'discrete_stress_distribution.png'))


if __name__ == '__main__':
    main()





