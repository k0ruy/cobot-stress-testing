# Script to tidy up the dataset for the project

# Libraries:
import os
import shutil
import zipfile
from pathlib import Path
import glob
from shutil import rmtree


# Function to unzip the dataset:
def unzip_data(data_dir: Path, zip_f: Path) -> None:
    """
    Unzip a zipped file into a directory.
    @param data_dir: The directory to unzip the file into.
    @param zip_f: The zipped file to unzip.
    :return: None
    """
    with zipfile.ZipFile(zip_f, "r") as zip_ref:
        zip_ref.extractall(data_dir)


def move_and_rename_files(data_dir: Path) -> None:
    """
    Rename the files in a directory, input format is id
    into the corresponding folders.
    @param data_dir: The directory to rename the files in.
    :return: None
    """
    # Change if necessary:
    types = ["cobot", "manual", "rest", "stroopeasy", "stroophard"]

    for t in types:
        if not os.path.exists(data_dir / t):
            os.makedirs(data_dir / t)

    file_path = Path(data_dir, "Data Cobot Experiment", "Signals")

    for filename in glob.iglob(str(file_path) + '**/*.txt', recursive=True):

        # correct the small inaccuracies:
        if "maual" in filename:
            os.rename(filename, filename.replace("maual", "manual"))
        if "stroop_hard" in filename:
            os.rename(filename, filename.replace("stroop_hard", "stroophard"))

        for t in types:
            if t in filename:
                new_filename = filename[:filename.index(t) + len(t)] + ".txt"
                os.rename(filename, new_filename)
                os.rename(new_filename, data_dir / t / new_filename.split(os.sep)[-1])


def delete_unnecessary_files(data_dir: Path) -> None:
    """
    Delete the unnecessary files.
    @param data_dir: The directory to delete the files in.
    :return: None
    """
    rmtree(data_dir / "Data Cobot Experiment")
    os.remove(data_dir / 'rest' / '06-05-30_09-41-40_rest.txt')


def splitter(data_dir: Path):
    """
    Split the data into patient folders.
    :param data_dir: directory to split the data into.
    :return: directory with patient folders.
    """
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if os.path.exists(Path(data_dir, curr_dirname)):
            shutil.rmtree(rf'{data_dir}' + '/' + curr_dirname)
        if curr_dirname != 'patient-08_2022':
            os.makedirs(Path(data_dir, curr_dirname))


def add_dir_to_saved_data(data_dir: Path):
    """
    Add the directory to the saved data.
    :param data_dir: directory to add the directory to.
    :return: None, saves the data.
    """
    for file in os.listdir(data_dir):

        if os.path.exists(Path('..','..', 'saved_data', str(file))):
            shutil.rmtree(r'../../saved_data/' + str(file))  # I know I didnt use Path()
        os.makedirs(Path('..', '..', 'saved_data', str(file)))


def file_mover(data_dir: Path):
    """
    Move the files to the correct folder.
    :param data_dir: the directory to move the files to.
    :return: None, moves the files.
    """
    for file in os.listdir(data_dir):
        patient_id = file.split('-')[0]
        curr_dirname = f'patient-{patient_id}'

        if file.startswith('08_2022'):
            shutil.move(Path(data_dir, str(file)), Path(data_dir, 'patient-08'))
        elif file.endswith('.txt'):
            shutil.move(Path(data_dir, str(file)), Path(data_dir, curr_dirname, str(file)))


def main(data_dir: Path, zip_f: Path) -> None:
    """
    Main function to run the script.
    @param data_dir: The directory to unzip the file into.
    @param zip_f: The zipped file to unzip.
    :return: None
    """
    # make sure the directory is clean and it exists:
    if os.path.exists(data_dir):
        rmtree(data_dir)
    os.makedirs(data_dir)

    # unzip the data:
    unzip_data(data_dir, zip_f)

    # move and rename the files:
    move_and_rename_files(data_dir)

    # clean up the directory:
    delete_unnecessary_files(data_dir)

    # split the data into patient folders:
    for exp in os.listdir(Path("..", "..", "experiments")):
        splitter(Path("..", "..", "experiments", str(exp)))
        file_mover(Path("..", "..", "experiments", str(exp)))
        add_dir_to_saved_data(Path("..", "..", "experiments", str(exp)))


# Driver:
if __name__ == "__main__":
    data_store = Path("..", "..", "experiments")
    zip_file = Path("..", "..", "Data Cobot Experiment.zip")
    main(data_store, zip_file)
