import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from src.utils.constants import get_local_validation_dir, get_small_local_validation_dir

local_val_dir = get_local_validation_dir()

train = pd.read_parquet(local_val_dir / "train.parquet")
all_train_unique_session = len(train.session.unique())

# last 7 days of data
seven_days = 7 * 24 * 60 * 60
train_cutoff = train.ts.max() - seven_days
train = train[train.ts > train_cutoff]
last_week_unique_session = len(train.session.unique())

# sample frac of session_id
frac = 0.35
lucky_sessions_train = train.drop_duplicates(["session"]).sample(frac=frac)["session"]
subset_of_train = train[train.session.isin(lucky_sessions_train)]
subset_last_week_unique_session = len(subset_of_train.session.unique())

del train
gc.collect()

print(f"train unique session: {all_train_unique_session}")
print(f"last week train unique session: {last_week_unique_session}")
print(f"subset last week train unique session: {subset_last_week_unique_session}")

train_features = []
train_labels = []

for grp in tqdm(subset_of_train.groupby("session")):
    # randomly select cutoff for label & features: sequence of aids
    if grp[1].shape[0] != 1:
        cutoff = np.random.randint(
            1, grp[1].shape[0]
        )  # we want at least a single item in our validation data for each session
        train_features.append(grp[1].iloc[:cutoff])
        train_labels.append(grp[1].iloc[cutoff:])

# save subset train & its label to small-local-validation
small_local_val_dir = get_small_local_validation_dir()
print("create df train feature")
train_features = pd.concat(train_features).reset_index(drop=True)
filepath = small_local_val_dir / "train.parquet"
print(f"save df train feature to: {filepath}")
train_features.to_parquet(filepath)

del train_features
gc.collect()

print("create df train ground truth")
train_labels = pd.concat(train_labels).reset_index(drop=True)
train_labels["type"] = train_labels["type"].map({0: "clicks", 1: "carts", 2: "orders"})
train_labels = (
    train_labels.groupby(["session", "type"])["aid"].apply(list).reset_index()
)
train_labels.columns = ["session", "type", "ground_truth"]
filepath = small_local_val_dir / "train_labels.parquet"
print(f"save df train feature to: {filepath}")
train_labels.to_parquet(filepath)

# # convert long format to session | list of ground truth for training
# train_labels.groupby("session")["aid"].apply(list)
