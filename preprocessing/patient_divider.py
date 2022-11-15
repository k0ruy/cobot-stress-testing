import shutil
from pathlib import Path
import os


def splitter(data_dir: Path):
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if os.path.exists(Path(data_dir, curr_dirname)):
            shutil.rmtree(rf'{data_dir}' + '/' + curr_dirname)
        if curr_dirname != 'patient-08_2022':
            os.makedirs(Path(data_dir, curr_dirname))


def add_dir_to_saved_data(data_dir: Path):
    for file in os.listdir(data_dir):

        if os.path.exists(Path('..', 'saved_data_test', str(file))):
            shutil.rmtree(r'../saved_data_test/' + str(file))  # I know I didnt use Path()
        os.makedirs(Path('..', 'saved_data_test', str(file)))


def file_mover(data_dir: Path):
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if file.startswith('08_2022'):
            shutil.move(Path(data_dir, str(file)), Path(data_dir, 'patient-08'))
        elif file.endswith('.txt'):
            shutil.move(Path(data_dir, str(file)), Path(data_dir, curr_dirname, str(file)))


if __name__ == '__main__':

    for exp in os.listdir(Path("..", "experiments")):
        splitter(Path("..", "experiments", str(exp)))
        file_mover(Path("..", "experiments", str(exp)))
        add_dir_to_saved_data(Path("..", "experiments", str(exp)))