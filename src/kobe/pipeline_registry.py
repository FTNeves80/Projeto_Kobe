"""Project pipelines."""
from __future__ import annotations

from kobe.pipelines.preparacaodados import pipeline as preparacao_pipeline
from kobe.pipelines.treinamento import pipeline as treinamento_pipeline
from kobe.pipelines.aplicacao import pipeline as aplicacao_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    pipelines = {
        "preparacao": preparacao_pipeline.create_pipeline(),
        "treinamento": treinamento_pipeline.create_pipeline(),
        "aplicacao": aplicacao_pipeline.create_pipeline(),
    }

    pipelines["__default__"] = preparacao_pipeline.create_pipeline() + treinamento_pipeline.create_pipeline() + aplicacao_pipeline.create_pipeline()

    return pipelines