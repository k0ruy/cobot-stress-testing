from rf_stress_prediction import *
from xgboost import XGBRegressor
from config import COBOT_RESULTS, MANUAL_RESULTS, MERGED_RESULTS


def xgboost_stress_predictions(task: str, type: str, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a XGBoost regression model on the stress prediction task.
    @param task: str, either 'cobot' or 'manual' or 'all'
    @param type: str, either 'ecg', 'eda' or 'emg'
    @param X_train: pandas DataFrame, training data
    @param X_test: pandas DataFrame, test data
    @param y_train: pandas Series, training labels
    @param y_test: pandas Series, test labels
    :return: None
    """
    # train:
    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)

    # evaluate:
    if task == 'cobot':
        path = COBOT_RESULTS
    elif task == 'manual':
        path = MANUAL_RESULTS
    else:
        path = MERGED_RESULTS

    results_path = Path(path, 'xgboost_regression')
    results_path.mkdir(parents=True, exist_ok=True)

    # save the test scores:
    r2 = np.round(r2_score(y_test, np.round(xgb.predict(X_test))), 3)
    test_scores = pd.DataFrame(r2, columns=['Score'], index=['R2'])
    test_scores.to_csv(results_path / f'{task}_{type}_test_scores.csv')

    # feature importances:
    feature_importances = pd.DataFrame(xgb.feature_importances_, index=X_train.columns, columns=['Importance'])
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    feature_importances.to_csv(results_path / f'{task}_{type}_feature_importances.csv')

    # sort the features by importance:
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    feature_importances[0:10].plot(kind='bar', figsize=(10, 5))
    # add space for the x-axis labels:
    plt.subplots_adjust(bottom=0.3)
    # rotate the x-axis labels:
    plt.xticks(rotation=15)
    plt.subplots_adjust(bottom=0.3)
    # no legend:
    plt.legend().set_visible(False)
    plt.title(f"Xgboost top 10 features for {task} {type} data")
    plt.savefig(Path(PLOTS, f"xgboost_{task}_{type}_feature_importance.png"))
    plt.show()


def main():

    for task in ['cobot', 'manual']:
        # load data:
        X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'ecg_{task}.csv')), add_demographics=False)

        X_train_eda, X_test_eda, y_train_eda, y_test_eda = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'eda_{task}.csv')), add_demographics=False)

        X_train_emg, X_test_emg, y_train_emg, y_test_emg = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'emg_{task}.csv')), add_demographics=False)

        # concatenate x and y train and test sets for total data prediction
        X_train_ecg_demo, X_test_ecg_demo, y_train_ecg_demo, y_test_ecg_demo = split_data(handle_missing_values(
            load_data(SAVED_DATA / f'ecg_{task}.csv')), add_demographics=True)
        X_train = pd.concat([X_train_ecg_demo, X_train_eda, X_train_emg], axis=1)
        X_test = pd.concat([X_test_ecg_demo, X_test_eda, X_test_emg], axis=1)
        y_train = y_train_ecg_demo  # all y_train sets are the same
        y_test = y_test_ecg_demo

        # train and evaluate:
        xgboost_stress_predictions(task, 'ecg', X_train_ecg, X_test_ecg, y_train_ecg, y_test_ecg)
        xgboost_stress_predictions(task, 'eda', X_train_eda, X_test_eda, y_train_eda, y_test_eda)
        xgboost_stress_predictions(task, 'emg', X_train_emg, X_test_emg, y_train_emg, y_test_emg)

        # remove duplicate columns:
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    xgboost_stress_predictions('all', 'merged', X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()
