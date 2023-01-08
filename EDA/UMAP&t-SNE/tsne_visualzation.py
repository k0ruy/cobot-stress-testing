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
    t_sne_path = SAVED_DATA / 'tsne_pca.csv'
    t_sne = pd.read_csv(t_sne_path)

    initializer = ""
    # check if the path has an initializer:
    if not t_sne_path.name[-7:-4] == 'sne':
        initializer = "_" + t_sne_path.name[-7:-4]

    nr_of_features = len(t_sne.columns)

    # load the target:
    target = pd.read_csv(SAVED_DATA / 'stress_for_tsne.csv')

    # create a dataframe with the target and the t-SNE features:
    df_subset = pd.DataFrame()
    # add the t-SNE features:
    for i in range(1, nr_of_features + 1):
        df_subset['tsne_' + '3' + '_f' + str(i)] = t_sne['tsne_' + '3' + '_f' + str(i)]
    # add the target:
    temp = target.Stress.reset_index()
    df_subset['y'] = temp.Stress.values.astype(int)

    # create a figure:
    fig = plt.figure(figsize=(16, 10))
    # create a 2D scatter plot:
    ax = fig.add_subplot(111)
    sc = ax.scatter(df_subset.tsne_3_f1, df_subset.tsne_3_f2, c=df_subset.y, s=20, alpha=0.8,
                    edgecolors='k', cmap='viridis')
    plt.title('t-SNE 2D Plot')
    plt.legend(title='Stress level', *sc.legend_elements())

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'tsne_2D.png'))
    except FileNotFoundError:
        # create the directory:
        Path(PLOTS / f't-SNE{initializer}').mkdir(parents=True, exist_ok=True)
        # save the figure:
        fig.savefig(Path(PLOTS / f't-SNE{initializer}', 'tsne_2D.png'))

    # create a figure:
    fig = plt.figure(figsize=(12, 8))

    # create a 3D scatter plot:
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df_subset.tsne_3_f1, df_subset.tsne_3_f2, df_subset.tsne_3_f3, c=df_subset.y, s=10, alpha=0.8,
                    edgecolors='k', cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('t-SNE 3D Plot')
    plt.legend(title='Stress level', *sc.legend_elements(), loc='upper right')

    # save the figure:
    try:
        fig.savefig(Path(PLOTS / 'tsne_3D.png'))
    except FileNotFoundError:
        # create the directory:
        Path(PLOTS / f't-SNE{initializer}').mkdir(parents=True, exist_ok=True)
        # save the figure:
        fig.savefig(Path(PLOTS / f't-SNE{initializer}', 'tsne_3D.png'))


# Driver:
if __name__ == '__main__':
    main()
