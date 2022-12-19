from pathlib import Path
import json
import os

ROOT_DIR = Path(__file__).parent.parent.parent


class CFG:
    N_test = 80
    N_train = 30
    covisit_candidates = 40
    fasttext_candidates = 20
    word2vec_candidates = 50
    matrix_factorization_candidates = 10


### Create directory


def check_directory(path: Path) -> None:
    # if the directory does not exist, create it
    if not os.path.exists(path):  # pragma: no cover
        os.makedirs(path)


### Create JSON file


def write_json(filepath: str, data: dict) -> None:
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)
        file.close()


### Data/Raw Dir


def get_local_validation_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "local-validation"
    check_directory(path)
    return path


def get_full_data_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "full-data"
    check_directory(path)
    return path


def get_scoring_covisitation_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "full-data" / "co-visitation"
    return path


def get_local_covisitation_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "local-validation" / "co-visitation"
    return path


def get_scoring_matrix_fact_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "full-data" / "matrix-factorization"
    return path


def get_local_matrix_fact_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "local-validation" / "matrix-factorization"
    return path


def get_scoring_word2vec_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "full-data" / "word2vec"
    return path


def get_local_word2vec_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "local-validation" / "word2vec"
    return path


def get_scoring_fasttext_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "full-data" / "fasttext"
    return path


def get_local_fasttext_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "local-validation" / "fasttext"
    return path


### Data/Processed Dir

### Data/Processed Dir: CANDIDATE RETRIEVAL


def get_processed_full_data_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "full-data"
    check_directory(path)
    return path


def get_processed_local_validation_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "local-validation"
    check_directory(path)
    return path


def get_processed_training_train_splitted_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_splitted"
    check_directory(path)
    return path


def get_processed_training_test_splitted_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_splitted"
    check_directory(path)
    return path


def get_processed_scoring_train_splitted_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_splitted"
    check_directory(path)
    return path


def get_processed_scoring_test_splitted_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_splitted"
    check_directory(path)
    return path


def get_processed_training_train_candidates_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_candidates"
    check_directory(path)
    return path


def get_processed_training_test_candidates_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_candidates"
    check_directory(path)
    return path


def get_processed_scoring_train_candidates_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_candidates"
    check_directory(path)
    return path


def get_processed_scoring_test_candidates_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_candidates"
    check_directory(path)
    return path


### Data/Processed Dir: FEATURES

### Data/Processed Dir: SESSION FEATURES


def get_processed_training_train_sess_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "session_features"
    check_directory(path)
    return path


def get_processed_training_test_sess_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "session_features"
    check_directory(path)
    return path


def get_processed_scoring_train_sess_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "session_features"
    check_directory(path)
    return path


def get_processed_scoring_test_sess_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "session_features"
    check_directory(path)
    return path


### Data/Processed Dir: SESSION-ITEM FEATURES


def get_processed_training_train_sess_item_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "session_item_features"
    check_directory(path)
    return path


def get_processed_training_test_sess_item_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "session_item_features"
    check_directory(path)
    return path


def get_processed_scoring_train_sess_item_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "session_item_features"
    check_directory(path)
    return path


def get_processed_scoring_test_sess_item_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "session_item_features"
    check_directory(path)
    return path


### Data/Processed Dir: ITEM FEATURES


def get_processed_training_train_item_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "item_features"
    check_directory(path)
    return path


def get_processed_training_test_item_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "item_features"
    check_directory(path)
    return path


def get_processed_scoring_train_item_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "item_features"
    check_directory(path)
    return path


def get_processed_scoring_test_item_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "item_features"
    check_directory(path)
    return path


### Data/Processed Dir: ITEM-HOUR FEATURES


def get_processed_training_train_item_hour_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "item_hour_features"
    check_directory(path)
    return path


def get_processed_training_test_item_hour_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "item_hour_features"
    check_directory(path)
    return path


def get_processed_scoring_train_item_hour_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "item_hour_features"
    check_directory(path)
    return path


def get_processed_scoring_test_item_hour_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "item_hour_features"
    check_directory(path)
    return path


### Data/Processed Dir: ITEM-WEEKDAY FEATURES


def get_processed_training_train_item_weekday_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "item_weekday_features"
    check_directory(path)
    return path


def get_processed_training_test_item_weekday_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "item_weekday_features"
    check_directory(path)
    return path


def get_processed_scoring_train_item_weekday_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "item_weekday_features"
    check_directory(path)
    return path


def get_processed_scoring_test_item_weekday_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "item_weekday_features"
    check_directory(path)
    return path


### Data/Processed Dir: ITEM-COVISITATION FEATURES


def get_processed_training_train_item_covisitation_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "item_covisitation_features"
    check_directory(path)
    return path


def get_processed_training_test_item_covisitation_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "item_covisitation_features"
    check_directory(path)
    return path


def get_processed_scoring_train_item_covisitation_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "item_covisitation_features"
    check_directory(path)
    return path


def get_processed_scoring_test_item_covisitation_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "item_covisitation_features"
    check_directory(path)
    return path


### Data/Processed Dir: SESSION REPRESENTATION ITEMS


def get_processed_training_train_session_representation_items_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "session_representation_items"
    check_directory(path)
    return path


def get_processed_training_test_session_representation_items_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "session_representation_items"
    check_directory(path)
    return path


def get_processed_scoring_train_session_representation_items_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "session_representation_items"
    check_directory(path)
    return path


def get_processed_scoring_test_session_representation_items_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "session_representation_items"
    check_directory(path)
    return path


### Data/Processed Dir: DISTANCE MATRIX FACTORIZATION FEATURES


def get_processed_training_train_matrix_fact_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "matrix_fact_features"
    check_directory(path)
    return path


def get_processed_training_test_matrix_fact_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "matrix_fact_features"
    check_directory(path)
    return path


def get_processed_scoring_train_matrix_fact_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "matrix_fact_features"
    check_directory(path)
    return path


def get_processed_scoring_test_matrix_fact_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "matrix_fact_features"
    check_directory(path)
    return path


### Data/Processed Dir: DISTANCE WORD2VEC FEATURES


def get_processed_training_train_word2vec_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "word2vec_features"
    check_directory(path)
    return path


def get_processed_training_test_word2vec_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "word2vec_features"
    check_directory(path)
    return path


def get_processed_scoring_train_word2vec_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "word2vec_features"
    check_directory(path)
    return path


def get_processed_scoring_test_word2vec_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "word2vec_features"
    check_directory(path)
    return path


### Data/Processed Dir: DISTANCE FASTTEXT FEATURES


def get_processed_training_train_fasttext_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_features" / "fasttext_features"
    check_directory(path)
    return path


def get_processed_training_test_fasttext_features_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_features" / "fasttext_features"
    check_directory(path)
    return path


def get_processed_scoring_train_fasttext_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_features" / "fasttext_features"
    check_directory(path)
    return path


def get_processed_scoring_test_fasttext_features_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_features" / "fasttext_features"
    check_directory(path)
    return path


### Data/Processed Dir: FINAL DATASET FEATURES


def get_processed_training_train_dataset_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "train_final_dataset"
    check_directory(path)
    return path


def get_processed_training_test_dataset_dir() -> Path:
    path = get_processed_local_validation_dir()
    path = path / "training" / "test_final_dataset"
    check_directory(path)
    return path


def get_processed_scoring_train_dataset_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "train_final_dataset"
    check_directory(path)
    return path


def get_processed_scoring_test_dataset_dir() -> Path:
    path = get_processed_full_data_dir()
    path = path / "scoring" / "test_final_dataset"
    check_directory(path)
    return path


### Output Dir: for scoring & submission


def get_data_output_dir() -> Path:
    path = ROOT_DIR / "data" / "output"
    check_directory(path)
    return path


def get_data_output_local_submission_dir(
    event: str, model: str, week_model: str
) -> Path:
    path = ROOT_DIR / "data" / "output" / "training" / event / week_model / model
    check_directory(path)
    return path


def get_data_output_submission_dir(event: str, model: str, week_model: str) -> Path:
    path = ROOT_DIR / "data" / "output" / "scoring" / event / week_model / model
    check_directory(path)
    return path


### Artifacts Dir: For training artifacts


def get_artifacts_dir() -> Path:
    path = ROOT_DIR / "artifacts"
    check_directory(path)
    return path


def get_artifacts_training_dir(week: str, event: str) -> Path:
    path = ROOT_DIR / "artifacts" / week / event
    check_directory(path)
    return path


def get_artifacts_tuning_dir(week: str, event: str) -> Path:
    path = ROOT_DIR / "artifacts" / "tuning" / week / event
    check_directory(path)
    return path


def get_artifacts_eda_dir(week: str, event: str) -> Path:
    path = ROOT_DIR / "artifacts" / "eda" / week / event
    check_directory(path)
    return path
