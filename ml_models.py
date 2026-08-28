# -*- coding: utf-8 -*-
"""Stable model-adapter boundary for the multi-page ML workspace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ml_engine import run_catboost_classification, run_catboost_regression
from random_forest_engine import (
    run_random_forest_classification,
    run_random_forest_regression,
)
from neural_network_engine import (
    run_neural_network_classification,
    run_neural_network_regression,
)


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    title: str
    family: str
    tasks: tuple[str, ...]
    status: str
    description: str


@dataclass(frozen=True)
class ModelAdapter:
    descriptor: ModelDescriptor
    runners: dict[str, Callable] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.runners)

    def run(self, frame, *, task="regression", **parameters):
        runner = self.runners.get(str(task or "regression"))
        if runner is None:
            if self.available:
                raise NotImplementedError(
                    f"Модель «{self.descriptor.title}» не поддерживает задачу «{task}»."
                )
            raise NotImplementedError(
                f"Модель «{self.descriptor.title}» ещё не подключена."
            )
        return runner(frame, **parameters)


MODEL_ADAPTERS = {
    "catboost": ModelAdapter(
        ModelDescriptor(
            key="catboost",
            title="CatBoost",
            family="Градиентный бустинг",
            tasks=("regression", "classification"),
            status="ready",
            description="Смешанные числовые и категориальные признаки без ручного кодирования.",
        ),
        runners={
            "regression": run_catboost_regression,
            "classification": run_catboost_classification,
        },
    ),
    "random-forest": ModelAdapter(
        ModelDescriptor(
            key="random-forest",
            title="Random Forest",
            family="Ансамбль деревьев",
            tasks=("regression", "classification"),
            status="ready",
            description="Устойчивый ансамбль деревьев с OOB-оценкой и контролем переобучения.",
        ),
        runners={
            "regression": run_random_forest_regression,
            "classification": run_random_forest_classification,
        },
    ),
    "neural-networks": ModelAdapter(
        ModelDescriptor(
            key="neural-networks",
            title="Нейросети",
            family="Глубокое обучение",
            tasks=("regression", "classification"),
            status="ready",
            description="Табличный MLP с масштабированием, early stopping и контролем переобучения.",
        ),
        runners={
            "regression": run_neural_network_regression,
            "classification": run_neural_network_classification,
        },
    ),
}


def get_model_adapter(key: str) -> ModelAdapter:
    try:
        return MODEL_ADAPTERS[str(key)]
    except KeyError as error:
        raise KeyError(f"Неизвестная ML-модель: {key}") from error
