from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent.parent


def check_directory(path: Path) -> None:
    # if the directory does not exist, create it
    if not os.path.exists(path):  # pragma: no cover
        os.makedirs(path)


def get_data_output_dir() -> Path:
    path = ROOT_DIR / "data" / "output"
    check_directory(path)
    return path


def get_local_validation_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "local-validation"
    check_directory(path)
    return path


def get_small_local_validation_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "small-local-validation"
    check_directory(path)
    return path


def get_full_data_dir() -> Path:
    path = ROOT_DIR / "data" / "raw" / "full-data"
    check_directory(path)
    return path


def get_covisitation_dir() -> Path:
    path = ROOT_DIR / "data" / "processed" / "co-visitation"
    return path


def get_artifacts_dir() -> Path:
    path = ROOT_DIR / "artifacts"
    check_directory(path)
    return path
