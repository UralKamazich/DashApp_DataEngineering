# -*- coding: utf-8 -*-
"""Stable model-adapter boundary for the multi-page ML workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ml_engine import run_catboost_regression


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
    runner: Callable | None = None

    @property
    def available(self) -> bool:
        return self.runner is not None

    def run(self, frame, **parameters):
        if self.runner is None:
            raise NotImplementedError(
                f"Модель «{self.descriptor.title}» ещё не подключена."
            )
        return self.runner(frame, **parameters)


MODEL_ADAPTERS = {
    "catboost": ModelAdapter(
        ModelDescriptor(
            key="catboost",
            title="CatBoost",
            family="Градиентный бустинг",
            tasks=("regression",),
            status="ready",
            description="Смешанные числовые и категориальные признаки без ручного кодирования.",
        ),
        runner=run_catboost_regression,
    ),
    "random-forest": ModelAdapter(
        ModelDescriptor(
            key="random-forest",
            title="Random Forest",
            family="Ансамбль деревьев",
            tasks=("regression", "classification"),
            status="planned",
            description="Устойчивый базовый алгоритм и независимое сравнение с бустингом.",
        ),
    ),
    "neural-networks": ModelAdapter(
        ModelDescriptor(
            key="neural-networks",
            title="Нейросети",
            family="Глубокое обучение",
            tasks=("regression", "classification"),
            status="planned",
            description="Отдельный контур подготовки, обучения и контроля переобучения.",
        ),
    ),
}


def get_model_adapter(key: str) -> ModelAdapter:
    try:
        return MODEL_ADAPTERS[str(key)]
    except KeyError as error:
        raise KeyError(f"Неизвестная ML-модель: {key}") from error

