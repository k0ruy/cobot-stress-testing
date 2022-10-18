import shutil
from pathlib import Path
import os


def splitter(data_dir: Path):
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if os.path.exists(Path(data_dir, curr_dirname)):
            shutil.rmtree(Path(data_dir, curr_dirname))
        os.makedirs(Path(data_dir, curr_dirname))


def file_mover(data_dir: Path):
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if file.endswith('.txt'):
            shutil.move(Path(data_dir, str(file)), Path(data_dir, curr_dirname, str(file)))


if __name__ == '__main__':

    for exp in os.listdir(Path("..", "experiments")):
        splitter(Path("..", "experiments", str(exp)))
        file_mover(Path("..", "experiments", str(exp)))
