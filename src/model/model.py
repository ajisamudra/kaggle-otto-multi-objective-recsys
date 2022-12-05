from abc import ABC, abstractmethod
import pandas as pd
from lightgbm import LGBMRanker, LGBMClassifier, log_evaluation, early_stopping
from catboost import CatBoostClassifier, CatBoostRanker, Pool

#### CLASSIFEIR MODEL


class ClassifierModel(ABC):
    @abstractmethod
    def fit(self, X_train, X_val, y_train, y_val):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    @abstractmethod
    def predict_proba(self, X_test):
        pass

    @abstractmethod
    def get_params(self):
        pass


class LGBClassifier(ClassifierModel):
    def __init__(self, **kwargs):
        self._early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        self._verbose = int(kwargs.pop("verbose", 100))

        self.feature_importances_ = 0
        self.best_score_ = 0

        kwargs["importance_type"] = "gain"

        # self._model = LGBMClassifier(**kwargs)
        self._model = LGBMClassifier(class_weight="balanced", **kwargs)
        self.hyperprams = {}

    def fit(self, X_train, X_val, y_train, y_val):
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["average_precision", "binary_logloss"],
            callbacks=[
                early_stopping(
                    stopping_rounds=self._early_stopping_rounds, verbose=self._verbose
                ),
                log_evaluation(100),
            ],
        )

        # feature importance as DataFrame
        self.feature_importances_ = pd.DataFrame(
            {
                "feature": X_train.columns.to_list(),
                "importance": self._model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False, ignore_index=True)

        # best_score as float
        self.best_score_ = float(
            list(self._model.best_score_.get("valid_0").values())[0]
        )

        self.hyperprams = self._model.get_params()

        return self

    def predict(self, X_test):
        result = self._model.predict_proba(
            X_test, num_iteration=self._model.best_iteration_
        )
        return result[:, 1]

    def predict_proba(self, X_test):
        return self._model.predict_proba(
            X_test, num_iteration=self._model.best_iteration_
        )

    def get_params(self):
        return self.hyperprams


class CatClassifier(ClassifierModel):
    def __init__(self, **kwargs):
        self._early_stopping_rounds = kwargs.get("early_stopping_rounds", 50)
        self._verbose = kwargs.pop("verbose", 100)

        self.feature_importances_ = None
        self.best_score_ = 0
        self.best_iteration = 0

        self._model = CatBoostClassifier(auto_class_weights="SqrtBalanced", **kwargs)
        self.hyperprams = {}

    def fit(self, X_train, X_val, y_train, y_val):
        self._model.fit(
            X=X_train,
            y=y_train,
            use_best_model=True,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=self._verbose,
        )

        # feature importance as DataFrame
        self.feature_importances_ = pd.DataFrame(
            {
                "feature": X_train.columns.to_list(),
                "importance": self._model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False, ignore_index=True)

        # best_score as float
        self.best_score_ = self._model.get_best_score()
        self.best_iteration = self._model.get_best_iteration()
        self.hyperprams = self._model.get_all_params()

        return self

    def predict(self, X_test):
        result = self._model.predict_proba(X_test, ntree_end=self.best_iteration)
        return result[:, 1]

    def predict_proba(self, X_test):
        return self._model.predict_proba(X_test, ntree_end=self.best_iteration)

    def get_params(self):
        return self.hyperprams

class EnsembleClassifierModels:
    """Wrapper for Ensemble Classifier Models
    It has list of trained models with different set of training data

    The final prediction score will be average of scores from the list of trained models
    """

    def __init__(self):
        self.list_models = []

    def append(self, model: ClassifierModel):
        self.list_models.append(model)
        return self

    def predict(self, X):
        # average of score from list_models
        y_preds = pd.DataFrame()
        for ix, model in enumerate(self.list_models):
            y_pred = model.predict(X)
            # convert np ndarray to pd Series
            y_pred = pd.Series(y_pred, name=f"y_pred_{ix}")
            y_preds = pd.concat([y_preds, y_pred], axis=1)
        y_preds = y_preds.mean(axis=1)
        return y_preds

    def predict_proba(self, X):
        # average of score from list_models
        y_preds = pd.DataFrame()
        for ix, model in enumerate(self.list_models):
            y_pred = model.predict_proba(X)
            # convert np ndarray to pd Series
            y_pred = pd.Series(y_pred, name=f"y_pred_{ix}")
            y_preds = pd.concat([y_preds, y_pred], axis=1)
        y_preds = y_preds.mean(axis=1)
        return y_preds


#### RANKER MODEL


class RankingModel(ABC):
    @abstractmethod
    def fit(self, X_train, X_val, y_train, y_val):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass


class LGBRanker(RankingModel):
    def __init__(self, **kwargs):
        self._early_stopping_rounds = kwargs.pop("early_stopping_rounds", 50)
        self._verbose = kwargs.pop("verbose", 100)

        self.feature_importances_ = None
        self.best_score_ = None
        self.best_iteration_ = None
        self.objective_ = None

        kwargs["importance_type"] = "gain"
        kwargs["objective"] = "rank_xendcg"

        self._model = LGBMRanker(**kwargs)
        # if self._apply_class_weight:
        # self._model = LGBMRanker(class_weight="balanced", **kwargs)

    def fit(self, X_train, X_val, y_train, y_val, group_train, group_val, eval_at):

        self._model.fit(
            X=X_train,
            y=y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            eval_at=eval_at,
            eval_metric=["map"],
            callbacks=[
                early_stopping(
                    stopping_rounds=self._early_stopping_rounds, verbose=self._verbose
                ),
                log_evaluation(10),
            ],
        )

        # feature importance as DataFrame
        self.feature_importances_ = pd.DataFrame(
            {
                "feature": X_train.columns.to_list(),
                "importance": self._model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False, ignore_index=True)

        # best_score as float
        self.best_score_ = float(
            list(self._model.best_score_.get("valid_0").values())[0]
        )
        self.best_iteration_ = self._model.best_iteration_
        self.objective_ = self._model.objective_

        return self

    def predict(self, X_test):
        return self._model.predict(X_test, num_iteration=self._model.best_iteration_)


class CATBRanker(RankingModel):
    def __init__(self, **kwargs):
        self._early_stopping_rounds = kwargs.pop("early_stopping_rounds", 50)
        self._verbose = kwargs.pop("verbose", 100)

        kwargs["custom_metric"] = ["MAP:top=12", "NDCG:top=12"]

        self.feature_importances_ = None
        self.best_score_ = None

        self._model = CatBoostRanker(**kwargs)

    def fit(self, X_train, X_val, y_train, y_val, group_train, group_val):

        train_pool = Pool(data=X_train, label=y_train, group_id=group_train)

        val_pool = Pool(data=X_val, label=y_val, group_id=group_val)

        self._model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=self._early_stopping_rounds,
            verbose=self._verbose,
        )

        # feature importance as DataFrame
        self.feature_importances_ = pd.DataFrame(
            {
                "feature": X_train.columns.to_list(),
                "importance": self._model.feature_importances_,
            }
        ).sort_values(by="importance", ascending=False, ignore_index=True)

        # best_score as float
        self.best_score_ = self._model.get_best_score()

        return self

    def predict(self, X_test):
        return self._model.predict(X_test)