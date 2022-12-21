import click
import joblib
import treelite
from src.model.model import (
    EnsembleModels,
)
from src.utils.constants import get_artifacts_training_dir

from src.utils.logger import get_logger

logging = get_logger()

TARGET = "label"


def export_treelite_model(
    artifact: str,
    event: str,
    week_model: str,
):
    artifact_path = get_artifacts_training_dir(event=event, week=week_model)
    # read artifact dir
    filepath = artifact_path / artifact
    logging.info(f"read artifacts model {event.upper()} from: {filepath}")
    model_name = f"{filepath}/model.pkl"
    ensemble_model: EnsembleModels = joblib.load(model_name)
    for j, model in enumerate(ensemble_model.list_models):
        single_model_name = f"{filepath}/lgbm_model_{j}.pkl"
        logging.info(f"export single lightgbm model {j}")
        joblib.dump(model._model, single_model_name)
        treelite_model = treelite.Model.load(single_model_name, model_format="lightgbm")
        treelite_model_name = f"{filepath}/treelite_lgbm_model_{j}.so"
        treelite_model.export_lib(
            toolchain="gcc", libpath=treelite_model_name, verbose=True
        )
        logging.info(f"exported! at {treelite_model_name}")


@click.command()
@click.option(
    "--artifact",
    default="2022-12-05_catboost_46313_82997",
    help="artifact folder for reading model.pkl",
)
@click.option(
    "--event",
    default="orders",
    help="avaiable event: clicks/carts/orders/all",
)
@click.option(
    "--week_model",
    default="w2",
    help="on which training data the model was trained, w1/w2; w1:scoring dir, w2:training dir",
)
def main(
    artifact: str,
    event: str,
    week_model: str = "w2",
):
    export_treelite_model(
        artifact=artifact,
        event=event,
        week_model=week_model,
    )


if __name__ == "__main__":
    main()
