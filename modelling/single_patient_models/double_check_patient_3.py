# Library to double check the results of patient 3:
from modelling.regression.rf_stress_prediction import handle_missing_values
from sklearn.ensemble import RandomForestRegressor
from modelling.single_patient_models.single_patient_rf_regression import load_patient_data, split_patient_data

if __name__ == '__main__':
    # Load the data:
    ecg, eda, emg = load_patient_data(3, 'cobot')

    # handle missing values
    ecg, eda, emg = handle_missing_values(ecg), handle_missing_values(eda), handle_missing_values(emg)


    # Split the data:
    x_train_ecg, x_test_ecg, y_train_ecg, y_test_ecg = split_patient_data(ecg)
    x_train_eda, x_test_eda, y_train_eda, y_test_eda = split_patient_data(eda)
    x_train_emg, x_test_emg, y_train_emg, y_test_emg = split_patient_data(emg)

    print(x_train_ecg.head())
    print(x_train_eda.head())
    print(x_train_emg.head())
    print(x_train_ecg.head())

    print(y_train_ecg.values)

    print(y_test_ecg.values)


    # train a random forest regressor for each of the three sensors:
    rf_ecg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_eda = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_emg = RandomForestRegressor(n_estimators=100, random_state=42)

    # fit the models:
    rf_ecg.fit(x_train_ecg, y_train_ecg)
    rf_eda.fit(x_train_eda, y_train_eda)
    rf_emg.fit(x_train_emg, y_train_emg)

    # evaluate the models
    print(rf_ecg.score(x_test_ecg, y_test_ecg))
    print(rf_eda.score(x_test_eda, y_test_eda))
    print(rf_emg.score(x_test_emg, y_test_emg))
