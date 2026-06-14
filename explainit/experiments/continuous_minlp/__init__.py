"""Continuous-target MINLP experiment.

Stages:

1. ``data_setup.py``        – download & preprocess datasets into ``data/<key>/``
2. ``model_setup.py``       – train a regression model per dataset into ``models/<key>/``
3. ``priority_sets.py``     – declarative priority sets, keyed by dataset/set
4. ``priorities_selection.py`` – workbench for inspecting coverage/plots/exemplars
5. ``minlp_test_config.yaml`` – per-experiment configuration
6. ``minlp_runner.py``      – run MINLP search using a priority set + config
7. ``random_runner.py``     – run random search using the same priority set + config

Outputs land under ``data/``, ``models/``, ``analysis/``, ``results/`` next
to this package.
"""
