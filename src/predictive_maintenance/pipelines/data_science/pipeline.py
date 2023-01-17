"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_costs_and_confusion, get_model_scores, train_node


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_node,
                inputs="master_table",
                outputs=["model", "x_valid", "y_valid", "model_params"],
                name="train_model",
            ),
            node(
                func=get_model_scores,
                inputs=["x_valid", "y_valid", "model"],
                outputs="metrics",
                name="get_scores",
            ),
            node(
                func=get_costs_and_confusion,
                inputs=["x_valid", "y_valid", "model"],
                outputs=["confusion", "costs"],
            ),
        ]
    )
