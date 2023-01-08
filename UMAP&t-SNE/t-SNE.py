# use t-SNE to reduce the dimensions of the dataset, both as a visualization tool
# and to check how the model performs with a smaller number of features
# Libraries:
# Data manipulation:

import pandas as pd
from pathlib import Path
from config import SAVED_DATA
from modelling.rf_stress_prediction import handle_missing_values

# Dimensionality reduction:
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Global variables:
# Dimension for the t-SNE:
nr_of_components: int = 3
# Initialization for the t-SNE:
initialization = 'pca'
perplexity: int = 40


def load_data(df):
    return pd.read_csv(df)


def merge_data(*args):
    return handle_missing_values(pd.concat(args))


# Driver:
if __name__ == '__main__':
    # merge all the dataset together
    ecg_c = SAVED_DATA / 'ecg_cobot.csv'
    ecg_m = SAVED_DATA / 'ecg_manual.csv'
    emg_c = SAVED_DATA / 'emg_cobot.csv'
    emg_m = SAVED_DATA / 'emg_manual.csv'
    eda_c = SAVED_DATA / 'eda_cobot.csv'
    eda_m = SAVED_DATA / 'eda_manual.csv'

    X = merge_data(load_data(ecg_m), load_data(ecg_c))
    X.to_csv(SAVED_DATA / 'merged_for_tsne.csv', index=False)
    # print(ecg_m.shape)
    print(X.head())
    print(X.shape)
    # save the CustomerId column for later use, drop it from the dataset:
    patient_id = X['patient']
    X.drop(['patient', 'filename', 'Task'], axis=1, inplace=True)

    # initialize the TSNE, since PCA is performing well. and it's advised in the documentation, we will
    # use the PCA initialization:
    tsne = TSNE(n_components=nr_of_components, init=initialization, verbose=1, perplexity=perplexity,
                learning_rate='auto')
    # tested perplexity based on this article: https://distill.pub/2016/misread-tsne/
    # tried 5, 30, 40 and 50 and N^0.5, where N is the number of samples, 40 gave the best visual results

    # fit
    tsne_results = tsne.fit_transform(X)

    # save the results as a dataframe, add the CustomerID as the first column:
    tsne_results_df = \
        pd.DataFrame(tsne_results, columns=[f'tsne_{nr_of_components}_f{i}'
                                            for i in range(1, nr_of_components + 1)])

    if initialization == "warn":
        # save the results:
        tsne_results_df.to_csv(SAVED_DATA / f'tsne.csv', index=False)
    else:
        # save the results:
        tsne_results_df.to_csv(SAVED_DATA / f'tsne_{initialization}.csv', index=False)
