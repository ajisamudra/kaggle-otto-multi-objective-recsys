import click
import pandas as pd
import polars as pl
import numpy as np
from tqdm import tqdm
import gc
from pathlib import Path
from src.utils.constants import (
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_sess_features_dir,
    get_processed_training_test_sess_features_dir,
    get_processed_scoring_train_sess_features_dir,
    get_processed_scoring_test_sess_features_dir,
)
from src.utils.logger import get_logger

logging = get_logger()

input_path = get_processed_scoring_train_candidates_dir()
cand_df = pl.read_parquet(f"{input_path}/train_0_clicks_rows.parquet")
cand_df.head()
