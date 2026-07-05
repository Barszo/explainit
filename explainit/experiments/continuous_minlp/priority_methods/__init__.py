"""Priority-branch experiment stage (MINLP + random-search baseline).

Self-contained pipeline that mirrors ``standard_methods/`` but drives the
priority-based explainers (``MINLSearchExplainer`` and
``RandomSearchExplainer``) against the declarative priority sets from
``priority_sets.py``. It writes the same result tables as the standard-methods
stage plus a per-counterfactual ``priority_score``.
"""
