"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

np.random.seed(42)


def _column_transformer():
    cat_selector = make_column_selector(dtype_exclude=np.number)
    num_selector = make_column_selector(dtype_include=np.number)

    return ColumnTransformer(
        transformers=[
            ("scaler", "passthrough", num_selector),
            ("cat_preprocessing", OrdinalEncoder(handle_unknown="error"), cat_selector),
        ],
        remainder="passthrough",
    )


def _param_grid() -> List[Dict[str, Any]]:
    "Pipe param grid"
    return [
        {
            "estimator": [
                LogisticRegression(
                    class_weight="balanced", solver="lbfgs", max_iter=500
                )
            ],
            "estimator__C": [0.01, 0.1, 0.25, 0.5, 1.0],
            "transformer__scaler": [StandardScaler()],
        },
        {
            "estimator": [RandomForestClassifier(n_estimators=500)],
            "estimator__max_depth": [8, 15, 30],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__class_weight": ["balanced", "balanced_subsample"],
        },
    ]


def _pipe():
    transformer = _column_transformer()
    return Pipeline([("transformer", transformer), ("estimator", "passthrough")])


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True)


def _get_search_cv():
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
    TARGET_COL = "failure_type"
    X, y = master_table.drop([TARGET_COL], axis=1), master_table[TARGET_COL]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y)

    search = _get_search_cv()
    search.fit(X_train, y_train)
    return search, X_valid, y_valid


def get_model_scores(X_valid: pd.DataFrame, y_valid: pd.Series, model: Pipeline):
    scores = cross_val_score(model, X_valid, y_valid, n_jobs=-1, error_score="raise", verbose=2, scoring=make_scorer(matthews_corrcoef))
    mean_, std_ = scores.mean(), scores.std()
    logging.info(f"MCC: {mean_:0.4f} (+/- {std_:0.4f})")
    return {
        "mcc_mean" : mean_,
        "mcc_std" : std_
    }
