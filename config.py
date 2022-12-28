# Project configuration file containg all the paths to the data and the saved data.
# Path: config.py
from pathlib import Path

# Project folders
ROOT = Path(__file__).parent

# Zipped datset:
ZIP_FILE = Path(ROOT, 'Data Cobot Experiment.zip')

# Data folders
EXPERIMENTS = Path(ROOT, 'experiments')
SAVED_DATA = Path(ROOT, 'saved_data')
QUESTIONNAIRES = Path(ROOT, 'questionnaires.pkl')
