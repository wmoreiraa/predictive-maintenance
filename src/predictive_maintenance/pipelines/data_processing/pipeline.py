"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import drop_cols, sanitize_col_names


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=sanitize_col_names,
                inputs="raw",
                outputs="raw_with_proper_names",
                name="sanitize_names",
            ),
            node(
                func=drop_cols,
                inputs="raw_with_proper_names",
                outputs="master_table",
                name="drop_cols",
            ),
        ]
    )
