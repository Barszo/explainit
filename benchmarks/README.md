# Benchmarks

`run_benchmark.py` measures the *quality* of the search methods on small
real-data tasks (scikit-learn datasets and models):

| Metric | Meaning |
| --- | --- |
| valid | Share of instances where the answer reaches the target **and** respects every constraint (allowed ranges, immutable features). |
| score | Mean preference score of the valid answers, relative to the best possible score (1.0). |
| model rows | Mean number of rows the model had to predict per instance (the main cost). |
| seconds | Mean wall-clock time per instance. |

```bash
# Run and print a table
uv run --group benchmark python benchmarks/run_benchmark.py

# Compare with the recorded baseline (baseline values in parentheses)
uv run --group benchmark python benchmarks/run_benchmark.py --compare benchmarks/results/baseline.json
```

On GitHub, run the **CI** workflow manually ("Run workflow") to get the same
comparison in the job summary.

`results/baseline.json` was recorded at commit `44e3dfd` (version 0.1.0),
before the refactoring. At that commit MINLP answers reach the target but
leave the allowed ranges, so its valid rate is 0%; see
`tests/test_known_bugs.py`.

When a change is meant to improve quality or speed, include the comparison
table in the pull request.
