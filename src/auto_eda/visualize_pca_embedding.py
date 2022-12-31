from sklearn.decomposition import PCA
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from src.utils.constants import (
    CFG,
    get_processed_training_train_splitted_dir,
    get_processed_training_test_splitted_dir,
    get_processed_scoring_train_splitted_dir,
    get_processed_scoring_test_splitted_dir,
    get_processed_training_train_candidates_dir,
    get_processed_training_test_candidates_dir,
    get_processed_scoring_train_candidates_dir,
    get_processed_scoring_test_candidates_dir,
)
from src.utils.word2vec import load_word2vec_embedding, load_word2vec_cbow_embedding
from src.utils.logger import get_logger

logging = get_logger()

input_path = get_processed_training_test_candidates_dir()
filepath = input_path / f"test_0_carts_rows.parquet"
df = pl.read_parquet(filepath)
df = df.with_columns([pl.col("label").sum().over("session").alias("n")])
active_users = df.filter(pl.col("n") > 5)
active_users = active_users["session"].unique()


word2vec_embedding = load_word2vec_embedding()
vectors = word2vec_embedding.vectors
print(f"embedding shape {vectors.shape}")
em_2d = PCA(n_components=2).fit_transform(vectors)
print(f"PCA embedding shape {em_2d.shape}")

# SELECT ONE USER WITH 20+ CLICKS
u = np.random.choice(active_users)
dff = df.filter(pl.col("session") == u).to_pandas()
tmp = dff[dff["label"] == 1]["candidate_aid"].values

############
## PLOT HISTORY BY ITEM CATEGORY
############

# PLOT CLICKS, CARTS, ORDERS OVER TSNE ITEM EMBEDDING PLOT
plt.figure(figsize=(15, 15))
plt.scatter(em_2d[::25, 0], em_2d[::25, 1], s=1, label="All 1.8M items!")
plt.plot(em_2d[tmp][:, 0], em_2d[tmp][:, 1], "-", color="orange")
plt.scatter(em_2d[tmp][:, 0], em_2d[tmp][:, 1], color="orange", s=25, label="Click")

# PLOT NUMBERS OF ORDER VISITED
old_xy = []
pos = []
for i, (x, y) in enumerate(zip(em_2d[tmp][:, 0], em_2d[tmp][:, 1])):
    new_location = True
    for j in old_xy:
        if (np.abs(x - j[0]) < 5) & (np.abs(y - j[1]) < 5):
            new_location = False
    if new_location:
        plt.text(x, y, f"{i+1}", size=18)
        old_xy.append((x, y))
        pos.append(i)

# LABEL PLOT
plt.legend()
plt.title(f"Test User {u} - {len(tmp)} labels", size=18)
# plt.xlabel('Item category',size=16)
plt.ylabel("\n\nItem category", size=16)
plt.xticks([], [])
plt.yticks([], [])
plt.show()
