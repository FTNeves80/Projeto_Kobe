"""
This is a boilerplate pipeline 'preparacaodados'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            nodes.preparacao,
            inputs=['raw_train'],
            outputs='data_filtered',
            tags=['preprocessamento']
        ),
        node(
            nodes.dividir_dados,
            inputs='data_filtered',
            outputs=['train_set', 'test_set'],
            tags=['split']
        ),
         node(
            nodes.treinar_modelo,
            inputs=['train_set', 'params:session_id'],
            outputs='best_model',
            tags=['train']
        ),
        node(
            nodes.treinar_lr,
            inputs=['train_set', 'params:session_id'],
            outputs='best_lr_model',
            tags=['train', 'logistic']
        )
    ])

