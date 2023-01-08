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
    umap_path = SAVED_DATA / 'umap.csv'
    umap = pd.read_csv(umap_path).values

    # load the target:
    target = pd.read_csv(SAVED_DATA / 'stress_for_umap.csv')

    c = target.Stress.values.astype(int)
    # create a figure:
    fig, ax = plt.subplots(1, figsize=(14, 10))
    sc = plt.scatter(*umap.T, c=c, s=20, alpha=0.8, edgecolors='k', cmap='viridis')
    plt.title('Fashion MNIST Embedded via UMAP')
    plt.legend(title='Stress level', *sc.legend_elements())

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'umap_2D.png'))
    except FileNotFoundError:
        # create the directory:
        Path(PLOTS / f'UMAP').mkdir(parents=True, exist_ok=True)
        # save the figure:
        fig.savefig(Path(PLOTS / f'UMAP', 'tsne_2D.png'))


# Driver:
if __name__ == '__main__':
    main()
