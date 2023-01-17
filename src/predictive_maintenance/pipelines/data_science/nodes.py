"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
from feature_engine.creation import MathFeatures
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

np.random.seed(42)


def _column_transformer() -> ColumnTransformer:
    """Create the column transformer steps that will be used

    Steps listed as "passthrough" with any names will be used in the CrossVal step.

    Returns:
        ColumnTransformer: scikit column transformer
    """
    cat_selector = make_column_selector(dtype_exclude=np.number)
    num_selector = make_column_selector(dtype_include=np.number)
    global temp_features
    temp_features = ["air_temperature_k", "process_temperature_k"]
    return ColumnTransformer(
        transformers=[
            ("scaler", "passthrough", num_selector),
            ("temperature", "passthrough", temp_features),
            ("cat_preprocessing", OrdinalEncoder(handle_unknown="error"), cat_selector),
        ],
        remainder="passthrough",
    )


def _param_grid() -> list[dict[str, Any]]:
    """Pipe param grid

    Returns:
        list[dict[str, Any]]: param grid used for CV
    """
    return [
        {
            "estimator": [
                LogisticRegression(
                    class_weight="balanced", solver="lbfgs", max_iter=1000
                )
            ],
            "estimator__C": [0.01, 0.1, 0.25, 0.5, 1.0],
            "transformer__scaler": [StandardScaler()],
            "transformer__temperature": [
                MathFeatures(variables=temp_features, func="sub"),
                "passthrough",
            ],
        },
        {
            "estimator": [RandomForestClassifier(n_estimators=500)],
            "estimator__max_depth": [8, 15, 30],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__class_weight": ["balanced", "balanced_subsample"],
            "transformer__temperature": [
                MathFeatures(variables=temp_features, func="sub"),
                "passthrough",
            ],
        },
    ]


def _pipe() -> Pipeline:
    """Scikit pipeline

    Returns:
        Pipeline:
    """
    transformer = _column_transformer()
    return Pipeline([("transformer", transformer), ("estimator", "passthrough")])


def _cv():
    """Split method"""
    return StratifiedKFold(n_splits=5, shuffle=True)


def _get_search_cv():
    """Search object with some metrics used as scorers"""
    pipe = _pipe()
    grid = _param_grid()
    cv = _cv()
    scoring = {
        "mcc": make_scorer(matthews_corrcoef),
        "f1": make_scorer(f1_score, average="weighted"),
        "acc": make_scorer(accuracy_score),
    }
    return GridSearchCV(
        estimator=pipe,
        n_jobs=-1,
        param_grid=grid,
        cv=cv,
        verbose=2,
        scoring=scoring,
        refit="mcc",
    )


def train_node(master_table: pd.DataFrame):
    """Training node also used to split validation dataset"""
    TARGET_COL = "failure_type"
    X, y = master_table.drop([TARGET_COL], axis=1), master_table[TARGET_COL]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.15, shuffle=True, stratify=y
    )
    search = _get_search_cv()
    search.fit(X_train, y_train)
    import pdb

    pdb.set_trace()
    rf_params = search.best_params_["estimator"].get_params()
    return search.best_estimator_, X_valid, y_valid, rf_params


def get_model_scores(X_valid: pd.DataFrame, y_valid: pd.Series, model: GridSearchCV):
    """Get model scores"""
    scores = cross_val_score(
        model,
        X_valid,
        y_valid,
        n_jobs=-1,
        error_score="raise",
        verbose=2,
        scoring=make_scorer(matthews_corrcoef),
    )
    mean_, std_ = scores.mean(), scores.std()
    logging.info(f"MCC: {mean_:0.4f} (+/- {std_:0.4f})")
    return {"mcc_mean": mean_, "mcc_std": std_}


def get_costs_and_confusion(x_valid: pd.DataFrame, y_valid: pd.Series, model: Pipeline):
    cm = ConfusionMatrixDisplay.from_estimator(
        model, x_valid, y_valid, xticks_rotation="vertical"
    )
    # If this is too slow could be changed to used numpy! (np.nditer)
    # I wont do it tought, for the sake of simplicity
    def sum_all_but_one_column(s: np.ndarray, i: int):
        return s[:i].sum() + s[i + 1 :].sum()

    error_count = []
    for i, row in enumerate(cm.confusion_matrix):
        error_count.append(sum_all_but_one_column(row, i))
    labels = cm.display_labels.tolist()
    error_count = dict(zip(labels, error_count, strict=True))

    return cm.figure_, error_count
