from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_and_save_score_distribution(
    dfs: List[pd.DataFrame], filepath: Path, dataset: str
):
    for i, df in enumerate(dfs):
        plt.figure(figsize=(6, 4))
        plt.title(f"score distribution per class")
        sns.histplot(
            x=df["y_hat"],
            hue=df["y"],
            element="step",
            stat="density",
            common_norm=False,
            bins=50,
        )
        plt.savefig(
            f"{filepath}/{dataset}_score_dist_per_class_{i}.png", bbox_inches="tight"
        )


def plot_and_save_feature_importance(
    df: pd.DataFrame, filepath: Path, metric: str = "importance"
):
    fig_path = f"{filepath}/feature_importances.png"
    csv_path = f"{filepath}/feature_importances.csv"
    n_feature = df.shape[0]

    plt.figure(figsize=(7, int(n_feature * 0.3)))
    plt.title("Feature Importances")
    sns.barplot(data=df, x=metric, y="feature", color="#348ABD")
    plt.savefig(fig_path, bbox_inches="tight")
    df.to_csv(csv_path, index=False)


def summarise_table(
    df: pd.DataFrame,
    key_col: str,
    other_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Computes min, median, max and std of columns with prefix in other_cols.

    :param df: Input DataFrame
    :param key_col: Name of the key column
    :param other_cols: Prefix in name of the columns to be summarised
    :return: dataframe containing the key and the summary columns
    """
    df = df.copy()
    for col_name in other_cols:
        df[f"min_{col_name}"] = df.iloc[
            :, [col.startswith(col_name) for col in df.columns]
        ].min(axis=1, skipna=False)
        df[f"median_{col_name}"] = df.iloc[
            :, [col.startswith(col_name) for col in df.columns]
        ].median(axis=1, skipna=True)
        df[f"max_{col_name}"] = df.iloc[
            :, [col.startswith(col_name) for col in df.columns]
        ].max(axis=1, skipna=False)
        df[f"std_{col_name}"] = df.iloc[
            :, [col.startswith(col_name) for col in df.columns]
        ].std(axis=1, skipna=True)

    columns_of_interest = [key_col] + [col for col in df if col.endswith(other_cols)]
    return df[columns_of_interest]


def summarise_feature_importance(
    importance_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Takes a list of feature importance dataframes from cross validation fits, summarise and return for artifacts.

    :param importance_dfs: list of dataframes containing feature names and importance values from cross validation fits
    :return: summary dataframe of feature importance containing mix, max, median, std
    """
    # rename columns and merge dfs
    renamed_dfs = []
    for ix, df in enumerate(importance_dfs):
        df = df.rename(
            columns={
                "feature": "feature",
                "importance": f"importance_{ix}",
            }
        ).set_index("feature")
        renamed_dfs.append(df)
    importance_df = pd.concat(renamed_dfs, axis=1, sort=True)
    # rename index as "feature" to handle index being unnamed
    importance_df = importance_df.rename_axis(index="feature").reset_index()

    # summarise df and sort by median
    importance_summary_df = summarise_table(importance_df, "feature", ("importance",))
    importance_summary_df = importance_summary_df.sort_values(
        "median_importance", ascending=False
    ).reset_index(drop=True)

    return importance_summary_df
