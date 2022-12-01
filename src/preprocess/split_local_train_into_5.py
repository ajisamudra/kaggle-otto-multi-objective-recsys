import pandas as pd
from src.utils.constants import get_local_validation_dir, get_small_local_validation_dir

local_val_dir = get_local_validation_dir()

train = pd.read_parquet(local_val_dir / "train.parquet")

# sample 20% of session_id
lucky_sessions_train = train.drop_duplicates(["session"]).sample(frac=0.2)["session"]
subset_of_train = train[train.session.isin(lucky_sessions_train)]

# save subset train to small-local-validation
small_local_val_dir = get_small_local_validation_dir()
subset_of_train.to_parquet(small_local_val_dir / "train.parquet")
