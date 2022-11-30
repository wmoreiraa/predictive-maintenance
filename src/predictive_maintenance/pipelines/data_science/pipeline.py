"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_node,
                inputs="master_table",
                outputs="fitted_model",
                name="train_model",
            )
        ]
    )
