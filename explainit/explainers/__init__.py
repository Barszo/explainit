"""Counterfactual search explainers."""

from explainit.explainers.minlp_search import MINLPSearchExplainer
from explainit.explainers.random_search import RandomSearchExplainer

__all__ = ["MINLPSearchExplainer", "RandomSearchExplainer"]


def __getattr__(name):
    if name == "MINLSearchExplainer":
        from explainit.explainers import minlp_search

        return minlp_search.__getattr__(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
