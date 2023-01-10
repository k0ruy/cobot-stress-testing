# Auxiliary file to produce a correlation matrix on the cobot and manual merged files.
# Data manipulation
import pandas as pd
import numpy as np
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables:
from config import SAVED_DATA, PLOTS


def main() -> None:
    # load data:
    df_stress = pd.read_csv(SAVED_DATA / 'cobot_manual_merged.csv')
    # remove questionnaire columns from the features, minus age, experience and sex:
    dropped_columns = ["ID", "Task", "File_number", "STAI_Total", "Fatigue", "Stress", "Mental_Demand",
                       "Physical_Demand", "Frustration", "NASA_Total", "PSS", "NARS_S1", "NARS_S2" ,
                       "NARS_S3", "discrete_stress"]

    X_stress = df_stress.drop(dropped_columns, axis=1)

    # Correlation matrix:
    corr_matrix = X_stress.corr()

    # Remove upper triangle of the matrix
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
    plt.title('Correlation matrix of the features in the COBOT and MANUAL datasets')
    plt.xticks([])
    plt.subplots_adjust(left=0.2)
    plt.savefig(PLOTS / 'correlation_matrix_cobot_manual.png', dpi=300)
    plt.show()

    # Next, select a threshold for the correlation coefficient, for example 0.8
    threshold = 0.8

    # Identify features that are highly correlated with each other
    to_drop = [column for column in corr_matrix.columns if any(abs(corr_matrix[column]) > threshold)]

    # print the features we want to drop:
    print(f"Count of features to drop based on a {threshold} threshold: {len(to_drop)}")

    """
    We have an extremely collinear dataset as we would expect since we are extracting a huge amount of features from
    just 3 signals.
    """


if __name__ == '__main__':
    main()

