"""Shared predictors; importing this package loads no model weights."""
from importlib import import_module

MODELS = ("vissalformer")


def load_predictor(name, **settings):
    if name not in MODELS:
        raise ValueError(f"Unknown predictor: {name}")
    return import_module(f"models.{name}").create_predictor(**settings)
