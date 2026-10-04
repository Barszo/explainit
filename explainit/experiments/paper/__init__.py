"""Publication studies for the continuous-target MINLP method.

This package is the *harness* for the paper: it does not define datasets,
models, priorities or search methods -- those live in
``explainit/experiments/continuous_minlp`` and are imported as a library.
What it adds is what a publication needs and the dev pipeline lacks:

* ``common/runspec.py``  -- expand a YAML grid into immutable, hashed run specs
* ``common/registry.py`` -- one immutable directory per run + a study index,
  so a sweep never overwrites its predecessor and can be resumed
* ``common/execute.py``  -- execute one run spec (one method, one config, one
  seed) and return per-counterfactual rows
* ``common/stats.py``    -- paired tests, effect sizes, bootstrap CIs
* ``common/plots.py``    -- sensitivity / Pareto / paired-delta figures

Studies:

* ``study1_hyperparams/`` -- hyperparameter selection and the
  priority-vs-budget Pareto front (badania I)
* ``study2_factors/``     -- effect of dataset size, noise, feature types, ...
  with statistical analysis (badania II)
* ``study3_comparison/``  -- comparison against the standard counterfactual
  methods on a matched feasible region and budget (badania III)

Run any study with::

    python -m explainit.experiments.paper.run_study --config <study>/config.yaml
"""
