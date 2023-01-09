# Project configuration file containing all the paths to the data and the saved data.
# Path: config.py
from pathlib import Path

# Project folders
ROOT = Path(__file__).parent

# Zipped dataset:
ZIP_FILE = Path(ROOT, 'Data Cobot Experiment.zip')

# Data folders
EXPERIMENTS = Path(ROOT, 'experiments')
SAVED_DATA = Path(ROOT, 'saved_data')
QUESTIONNAIRES = Path(ROOT, 'questionnaires.pkl')
QUESTIONNAIRES_CLEANED = Path(SAVED_DATA, 'questionnaires_cleaned.pkl')

# Results folders
RESULTS = Path(ROOT, 'results')
COBOT_RESULTS = Path(RESULTS, 'cobot_results')
MANUAL_RESULTS = Path(RESULTS, 'manual_results')
PLOTS = Path(RESULTS, 'plots')
