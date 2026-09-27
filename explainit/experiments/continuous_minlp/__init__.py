"""Continuous-target MINLP experiment.

Stages:

1. ``data_setup.py``        – download & preprocess datasets into ``data/<key>/``
2. ``model_setup.py``       – train a regression model per dataset into ``models/<key>/``
3. ``priority_sets.py``     – declarative priority sets, keyed by dataset/set
4. ``priorities_selection.py`` – workbench for inspecting coverage/plots/exemplars
5. ``priority_methods/``    – MINLP + random-search baseline on the priority sets
6. ``standard_methods/``    – well-known regression counterfactual baselines
7. ``legacy/``              – superseded per-pair JSON runners, kept for reference

Publication studies (hyperparameter sweeps, factor studies, method
comparison) live in ``explainit/experiments/paper`` and import the stages
above as a library.

Outputs land under ``data/``, ``models/``, ``analysis/`` and each branch's
own ``results/`` directory.
"""
