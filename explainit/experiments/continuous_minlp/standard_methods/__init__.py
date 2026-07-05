"""Standard-methods stage for the continuous-target counterfactual experiment.

This package sits on top of the existing ``data_setup`` / ``model_setup``
stages. It:

1. builds a *predicted-target* dataset (target replaced by model output),
2. lets you explore it in a small notebook,
3. selects samples + targets and per-feature actionability from a config,
4. runs a battery of standard/regression counterfactual methods, and
5. persists linked result tables (samples, counterfactuals, metrics, summary).
"""
