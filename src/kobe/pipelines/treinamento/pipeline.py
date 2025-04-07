"""
This is a boilerplate pipeline 'treinamento'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=nodes.treinar,
            inputs=["train_set", "params:session_id"],
            outputs=None,
            tags=["train"]
        )

    ])
