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
    # load the t-SNE dataset:
    umap_path = SAVED_DATA / 'cobot_manual_umap.csv'
    umap = pd.read_csv(umap_path).values

    # 2D and 3D datasetes:
    # rename the columns to umap_2_f1, umap_2_f2, umap_3_f1, umap_3_f2, umap_3_f3
    umap_2 = umap[:, :2]
    umap_3 = umap[:, :3]
    umap_2 = pd.DataFrame(umap_2, columns=['umap_2_f1', 'umap_2_f2'])
    umap_3 = pd.DataFrame(umap_3, columns=['umap_3_f1', 'umap_3_f2', 'umap_3_f3'])

    # load the target:
    target = pd.read_csv(SAVED_DATA / 'stress_for_umap.csv', index_col=0)

    c = target.Stress.values.astype(int)
    # create a 2D figure:
    fig, ax = plt.subplots(1, figsize=(14, 10))
    sc = ax.scatter(umap_2.umap_2_f1, umap_2.umap_2_f2, c=c, s=20, alpha=0.8,
                    edgecolors='k', cmap='viridis')
    plt.title('Stress embedded via UMAP 2D')

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'umap_2D.png'))
    except FileNotFoundError:
        # create the directory:
        Path(PLOTS / f'UMAP').mkdir(parents=True, exist_ok=True)
        # save the figure:
        fig.savefig(Path(PLOTS / f'UMAP', 'umap_2D.png'))


    # create a 3D figure:
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(umap_3.umap_3_f1, umap_3.umap_3_f2, umap_3.umap_3_f3, c=c, s=20, alpha=0.8,
                    edgecolors='k', cmap='viridis')
    plt.title('Stress Embedded via UMAP 3D')
    plt.legend(title='Stress level', *sc.legend_elements())

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'umap_3D.png'))
    except FileNotFoundError:
        # create the directory:
        Path(PLOTS / f'UMAP').mkdir(parents=True, exist_ok=True)
        # save the figure:
        fig.savefig(Path(PLOTS / f'UMAP', 'umap_3D.png'))


# Driver:
if __name__ == '__main__':
    main()
