# Library to use LDA for dimensionality reduction and check if the tasks:
# cobot, manual, rest, stroopeasy and stroophard
# are separable given the features we extracted from the ECG, EDA and EMG signals.
import numpy as np
# Data manipulation:
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Modelling:
from modelling.regression.rf_stress_prediction import handle_missing_values, split_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# Plotting:
import matplotlib.pyplot as plt

# Global variables:
from config import SAVED_DATA, PLOTS


def import_all_merged_datasets() -> pd.DataFrame:
    """
    Function to import the datasets for all tasks
    :return:
    """
    # Cobot:
    ecg_cobot = pd.read_csv(SAVED_DATA / 'ecg_cobot.csv')
    eda_cobot = pd.read_csv(SAVED_DATA / 'eda_cobot.csv')
    emg_cobot = pd.read_csv(SAVED_DATA / 'emg_cobot.csv')

    cobot = pd.concat([ecg_cobot, eda_cobot, emg_cobot], axis=1)

    # Manual:
    ecg_manual = pd.read_csv(SAVED_DATA / 'ecg_manual.csv')
    eda_manual = pd.read_csv(SAVED_DATA / 'eda_manual.csv')
    emg_manual = pd.read_csv(SAVED_DATA / 'emg_manual.csv')

    manual = pd.concat([ecg_manual, eda_manual, emg_manual], axis=1)

    # Rest:
    ecg_rest = pd.read_csv(SAVED_DATA / 'ecg_rest.csv')
    eda_rest = pd.read_csv(SAVED_DATA / 'eda_rest.csv')
    emg_rest = pd.read_csv(SAVED_DATA / 'emg_rest.csv')

    rest = pd.concat([ecg_rest, eda_rest, emg_rest], axis=1)

    # Stroop easy:
    ecg_stroopeasy = pd.read_csv(SAVED_DATA / 'ecg_stroopeasy.csv')
    eda_stroopeasy = pd.read_csv(SAVED_DATA / 'eda_stroopeasy.csv')
    emg_stroopeasy = pd.read_csv(SAVED_DATA / 'emg_stroopeasy.csv')

    stroopeasy = pd.concat([ecg_stroopeasy, eda_stroopeasy, emg_stroopeasy], axis=1)

    # Stroop hard:
    ecg_stroophard = pd.read_csv(SAVED_DATA / 'ecg_stroophard.csv')
    eda_stroophard = pd.read_csv(SAVED_DATA / 'eda_stroophard.csv')
    emg_stroophard = pd.read_csv(SAVED_DATA / 'emg_stroophard.csv')

    stroophard = pd.concat([ecg_stroophard, eda_stroophard, emg_stroophard], axis=1)

    # add Task column to rest, stroopeasy and stroophard:
    rest['Task'] = 'REST'
    stroopeasy['Task'] = 'STROOPEASY'
    stroophard['Task'] = 'STROOPHARD'

    # drop filename and patient from all datasets:
    cobot = cobot.drop(['filename', 'patient'], axis=1)
    manual = manual.drop(['filename', 'patient'], axis=1)
    rest = rest.drop(['filename', 'patient'], axis=1)
    stroopeasy = stroopeasy.drop(['filename', 'patient'], axis=1)
    stroophard = stroophard.drop(['filename', 'patient'], axis=1)

    # handle missing values:
    cobot = handle_missing_values(cobot)
    manual = handle_missing_values(manual)
    rest = handle_missing_values(rest)
    stroopeasy = handle_missing_values(stroopeasy)
    stroophard = handle_missing_values(stroophard)

    # concatenate all datasets:
    df_other = pd.concat([rest, stroopeasy, stroophard], axis=0)
    df_task = pd.concat([cobot, manual], axis=0)

    # remove duplicate columns:
    df_other = df_other.loc[:, ~df_other.columns.duplicated()]
    df_task = df_task.loc[:, ~df_task.columns.duplicated()]

    # save partial merged datasets:
    df_task.to_csv(SAVED_DATA / 'cobot_manual_merged.csv', index=False)
    df_other.to_csv(SAVED_DATA / 'rest_stroopeasy_stroophard_merged.csv', index=False)

    # merge all datasets:
    df = pd.concat([df_task, df_other], axis=0)

    # drop columns with missing values:
    df = df.dropna(axis=1)

    # save the merged dataset:
    df.to_csv(SAVED_DATA / 'all_merged.csv', index=False)

    return df


def main() -> None:
    # load data:
    df_task = import_all_merged_datasets()

    X_task = df_task.drop(['Task'], axis=1)
    y_task = df_task['Task']

    # generate the model:
    lda = LinearDiscriminantAnalysis()

    # Fit the model to the data, no need to split since we are doing visualisations at this stage.
    lda.fit(X_task, y_task)

    # Transform the data to the new LDA space
    X_lda = lda.transform(X_task)

    # map y to 0, 1, 2, 3, 4:
    y = y_task.map({'COBOT': 0, 'MANUAL': 1, 'REST': 2, 'STROOPEASY': 3, 'STROOPHARD': 4})

    # assign the colors to each task:
    task_colors = ['red', 'blue', 'green', 'orange', 'purple']
    # Plot the clusters identified by the model
    plt.figure(figsize=(10, 10))
    plt.scatter(X_lda[y == 0, 0], X_lda[y == 0, 1], color=task_colors[0], label='COBOT', s=10, alpha=0.5)
    plt.scatter(X_lda[y == 1, 0], X_lda[y == 1, 1], color=task_colors[1], label='MANUAL', s=10, alpha=0.5)
    plt.scatter(X_lda[y == 2, 0], X_lda[y == 2, 1], color=task_colors[2], label='REST', s=10, alpha=0.5)
    plt.scatter(X_lda[y == 3, 0], X_lda[y == 3, 1], color=task_colors[3], label='STROOPEASY', s=10, alpha=0.5)
    plt.scatter(X_lda[y == 4, 0], X_lda[y == 4, 1], color=task_colors[4], label='STROOPHARD', s=10, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Task performed')
    plt.subplots_adjust(right=0.8)
    plt.title('LDA Clusters by patient task')
    plt.savefig(PLOTS / 'lda_clusters_tasks.png', dpi=300)
    plt.show()

    # LDA on the COBOT and MANUAL stress levels:
    # load data:
    df_stress = pd.read_csv(SAVED_DATA / 'cobot_manual_merged.csv')
    # remove questionnaire columns from the features, minus age, experience and sex:
    dropped_columns = ["ID", "Task", "File_number", "STAI_Total", "Fatigue", "Stress", "Mental_Demand",
                       "Physical_Demand", "Frustration", "NASA_Total", "PSS", "NARS_S1", "NARS_S2" ,
                       "NARS_S3", "discrete_stress"]

    X_stress = df_stress.drop(dropped_columns, axis=1)
    y_stress = df_stress['discrete_stress']

    # generate the model:
    lda_stress = LinearDiscriminantAnalysis()

    # fit the model:
    lda_stress.fit(X_stress, y_stress)

    # Transform the data to the new LDA space
    X_lda_stress = lda_stress.transform(X_stress)

    # assign the colors to each stress level:
    stress_colors = ['red', 'blue', 'green']

    # Plot the clusters identified by the model
    plt.figure(figsize=(10, 10))
    plt.scatter(X_lda_stress[y_stress == 0, 0], X_lda_stress[y_stress == 0, 1], color=stress_colors[0], label='Low stress', s=10, alpha=0.5)
    plt.scatter(X_lda_stress[y_stress == 1, 0], X_lda_stress[y_stress == 1, 1], color=stress_colors[1], label='Medium stress', s=10, alpha=0.5)
    plt.scatter(X_lda_stress[y_stress == 2, 0], X_lda_stress[y_stress == 2, 1], color=stress_colors[2], label='High stress', s=10, alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Stress level')
    plt.subplots_adjust(right=0.8)
    plt.title('LDA Clusters by stress level with COBOT and MANUAL data')
    plt.savefig(PLOTS / 'lda_clusters_stress.png', dpi=300)
    plt.show()
    plt.close()

    """
    Linear clustering works reasonably well on the tasks, it fails on the stress levels, generated with a uniform
    strategy, so stress levels are not linearly separable, so we tried quadratic LDA:
    """

    # Quadratic LDA on the COBOT and MANUAL stress levels:
    # generate the model:
    qda = QuadraticDiscriminantAnalysis()

    # fit the model:
    qda.fit(X_stress, y_stress)

    # predict the classes:
    y_pred = qda.predict(X_stress)

    # labels:
    labels = np.where(y_pred == 0, 'Low stress', np.where(y_pred == 1, 'Medium stress', np.where(y_pred == 2,
                                                                                                 'High stress')))

    # Plot the clusters identified by the model
    plt.figure(figsize=(10, 10))
    plt.scatter(X_stress.iloc[:, 0], X_stress.iloc[:, 1], c=y_pred, cmap='viridis', s=10, alpha=0.5)
    plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Stress level')
    plt.subplots_adjust(right=0.8)
    plt.title('QDA Clusters by stress level with COBOT and MANUAL data')
    plt.savefig(PLOTS / 'qda_clusters_stress.png', dpi=300)
    plt.show()
    plt.close()

    """
    QDA also does not separate the clusters well visually, so we tried T-sne and UMAP.
    """


if __name__ == '__main__':
    main()

