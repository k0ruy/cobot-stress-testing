# Auxiliary library to visualize the t-SNE 3D plot
# Libraries:
# Data Manipulation:
from pathlib import Path
import pandas as pd
from config import SAVED_DATA, PLOTS
from sklearn.preprocessing import LabelEncoder

# Visualization:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
mpl.use('tkagg')


# Driver:
if __name__ == '__main__':
    # load the t-SNE dataset:
    t_sne_path = SAVED_DATA / 'tsne_pca.csv'
    t_sne = pd.read_csv(t_sne_path, index_col=0)

    initializer = ""
    # check if the path has an initializer:
    if not t_sne_path.name[-7:-4] == 'sne':
        initializer = "_" + t_sne_path.name[-7:-4]

    nr_of_features = len(t_sne.columns)
    print(t_sne.columns)

    # load the target:
    target = pd.read_csv(SAVED_DATA / 'merged_for_tsne.csv', index_col=0)

    # create a dataframe with the target and the t-SNE features:
    df_subset = pd.DataFrame()
    # add the t-SNE features:
    for i in range(2, nr_of_features + 1):
        df_subset['tsne_' + '3' + '_f' + str(i)] = t_sne['tsne_' + '3' + '_f' + str(i)]
    # add the target:
    e = LabelEncoder()
    df_subset['y'] = e.fit_transform(target['Task'])
    df_subset['y'] = df_subset.y.astype(int)
    df_subset.reset_index(inplace=True)
    print(df_subset)

    if nr_of_features in (2, 3):
        # create a figure:
        fig = plt.figure(figsize=(16, 10))
        # create a 2D scatter plot:
        sns.scatterplot(x=f'tsne_3_f1', y=f'tsne_3_f2',
                        hue='y', data=df_subset, cmap=plt.cm.rainbow, s=20, alpha=0.8, edgecolors='k')
        plt.title('t-SNE 2D Plot')

        # save the figure:
        try:
            fig.savefig(Path(PLOTS / 'online_sales_dataset_dr_tsne_2D.png'))
        except FileNotFoundError:
            # create the directory:
            Path(PLOTS / f't-SNE{initializer}').mkdir(parents=True, exist_ok=True)
            # save the figure:
            fig.savefig(Path(PLOTS / f't-SNE{initializer}', 'online_sales_dataset_dr_tsne_2D.png'))

