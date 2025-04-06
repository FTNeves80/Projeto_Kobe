"""
This is a boilerplate pipeline 'aplicacao'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.inferencia,
            inputs=['raw_prod'],
            outputs="predictions",
            tags=['inferencia']
        )
    ])