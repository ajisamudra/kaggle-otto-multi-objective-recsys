from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent.parent


def check_directory(path: Path) -> None:
    # if the directory does not exist, create it
    if not os.path.exists(path):  # pragma: no cover
        os.makedirs(path)


### Data/Raw Dir


def get_local_validation_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "local-validation"
    check_directory(path)
    return path


def get_full_data_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "full-data"
    check_directory(path)
    return path


def get_covisitation_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "co-visitation"
    return path


### Data/Processed Dir


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


### Output & Artifacts Dir


def get_data_output_dir() -> Path:
    path = ROOT_DIR / "data" / "output"
    check_directory(path)
    return path


def get_artifacts_dir() -> Path:
    path = ROOT_DIR / "artifacts"
    check_directory(path)
    return path
