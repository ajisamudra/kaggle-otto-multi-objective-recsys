import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    get_processed_local_validation_dir,
    get_processed_full_data_dir,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
)
from src.utils.logger import get_logger

logging = get_logger()
