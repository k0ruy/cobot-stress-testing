# Auxiliary library to visualize the t-SNE 3D plot
# Libraries:
# Data Manipulation:
from pathlib import Path
import pandas as pd
from config import SAVED_DATA, PLOTS

# Visualization:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.use('tkagg')


def main() -> None:
    """
    Main function to run the script
    :return: None. Saves the plots to the plots folder
    """
    # load the t-SNE datasets:
    cmr_umap_path = SAVED_DATA / 'cobot_manual_umap_regression.csv'
    cmc_umap_path = SAVED_DATA / 'cobot_manual_umap_classification.csv'
    cr_umap_path = SAVED_DATA / 'cobot_umap_regression.csv'
    cc_umap_path = SAVED_DATA / 'cobot_umap_classification.csv'
    mr_umap_pat = SAVED_DATA / 'manual_umap_regression.csv'
    mc_umap_path = SAVED_DATA / 'manual_umap_classification.csv'
    paths = [cmr_umap_path, cmc_umap_path, cr_umap_path, cc_umap_path, mr_umap_pat, mc_umap_path]

    # load the dataframes:
    dataframes = [pd.read_csv(path) for path in paths]
    # names of the dataframes:
    names = ['cobot_manual_regression', 'cobot_manual_classification', 'cobot_regression', 'cobot_classification',
             'manual_regression', 'manual_classification']

    target_csv_names = ['cobot_manual', 'cobot_manual', 'cobot', 'cobot', 'manual', 'manual']

    dataframes = [df.rename(columns={'0': 'umap_f1', '1': 'umap_f2', '2': 'umap_f3'}) for df in dataframes]

    # create a 2D figure for each dataset:
    for i, df in enumerate(dataframes):

        # load the target:
        target = pd.read_csv(SAVED_DATA / f'{target_csv_names[i]}_stress_for_umap.csv', index_col=0)
        c = target.Stress.values.astype(int)

        # create a figure:
        fig = plt.figure(figsize=(16, 10))
        # create a 2D scatter plot:
        ax = fig.add_subplot(111)
        sc = ax.scatter(df.umap_f1, df.umap_f2, c=c, s=20, alpha=0.8,
                        edgecolors='k', cmap='viridis')
        plt.title(f'UMAP 2D Plot for {names[i]}')
        plt.legend(title='Stress level', *sc.legend_elements())

        # save the figure:
        try:
            fig.savefig(Path(PLOTS / 'UMAP', f'umap_2D_{names[i]}.png'))
        except FileNotFoundError:
            # create the directory:
            Path(PLOTS / f'UMAP').mkdir(parents=True, exist_ok=True)
            # save the figure:
            fig.savefig(Path(PLOTS / f'UMAP', f'umap_2D_{names[i]}.png'))

    # create a 3D figure for each dataset:
    for i, df in enumerate(dataframes):
        # load the target:
        target = pd.read_csv(SAVED_DATA / f'{target_csv_names[i]}_stress_for_umap.csv', index_col=0)
        c = target.Stress.values.astype(int)
        # create a figure:
        fig = plt.figure(figsize=(16, 10))
        # create a 3D scatter plot:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(df.umap_f1, df.umap_f2, df.umap_f3, c=c, s=20, alpha=0.8,
                        edgecolors='k', cmap='viridis')
        plt.title(f'UMAP 3D Plot for {names[i]}')
        plt.legend(title='Stress level', *sc.legend_elements())

        # save the figure:
        try:
            fig.savefig(Path(PLOTS / 'UMAP', f'umap_3D_{names[i]}.png'))
        except FileNotFoundError:
            # create the directory:
            Path(PLOTS / f'UMAP').mkdir(parents=True, exist_ok=True)
            # save the figure:
            fig.savefig(Path(PLOTS / f'UMAP', f'umap_3D_{names[i]}.png'))


# Driver:
if __name__ == '__main__':
    main()
