import numpy as np
from conftest import linear_model, mixed_model

from explainit.explainers.minlp_search import MINLPSearchExplainer
from explainit.explainers.random_search import RandomSearchExplainer
from explainit.utils import priority_plots


def _pngs(directory):
    return sorted(p.name for p in directory.glob("*.png"))


def test_plot_priorities_writes_one_file_per_actionable_entry(mixed_priorities, tmp_path):
    written = priority_plots.plot_priorities(
        mixed_priorities,
        sample=[0.1, 0.2, 0, 1],
        exemplar=[0.6, 0.7, 1, 0],
        feature_names=["income", "savings", "owner", "renter"],
        save_dir=tmp_path,
    )
    assert len(written) == 3
    assert all(path.exists() for path in written)


def test_plot_priorities_without_save_dir_writes_nothing(mixed_priorities, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert priority_plots.plot_priorities(mixed_priorities) == []
    assert list(tmp_path.iterdir()) == []


def test_plot_priorities_with_target_distribution(mixed_priorities, mixed_dataset, tmp_path):
    written = priority_plots.plot_priorities(
        mixed_priorities,
        sample=mixed_dataset[0],
        feature_matrix=mixed_dataset,
        target_values=mixed_model(mixed_dataset),
        target_threshold=0.9,
        save_dir=tmp_path,
    )
    assert len(written) == 3


def test_plot_probability_distributions(mixed_priorities, tmp_path):
    written = priority_plots.plot_probability_distributions(
        mixed_priorities, n_samples=300, save_dir=tmp_path
    )
    assert len(written) == 3


def test_sample_numeric_value_stays_in_bounds(numerical_priorities):
    rng = np.random.RandomState(0)
    constraint = numerical_priorities["numerical"][0]
    values = [priority_plots.sample_numeric_value(constraint, rng=rng) for _ in range(200)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert np.mean(values) > 0.5  # increasing priority favours high values


def test_random_search_plot_methods(numerical_priorities, tmp_path):
    explainer = RandomSearchExplainer(linear_model, numerical_priorities, np.array([0.2, 0.4]), 0.9)
    explainer.display_priorities(save_dir=tmp_path, show=False)
    explainer.investigate_probability_distribution(
        n_samples=300, save_dir=tmp_path / "dist", show=False
    )
    assert len(_pngs(tmp_path)) == 2
    assert len(_pngs(tmp_path / "dist")) == 2


def test_random_search_plots_write_no_files_by_default(numerical_priorities, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    explainer = RandomSearchExplainer(linear_model, numerical_priorities, np.array([0.2, 0.4]), 0.9)
    explainer.display_priorities()
    explainer.investigate_probability_distribution(n_samples=100)
    assert list(tmp_path.iterdir()) == []


def test_minlp_display_priorities(mixed_priorities, mixed_dataset, tmp_path):
    explainer = MINLPSearchExplainer(
        mixed_model, mixed_priorities, np.array([0.1, 0.2, 0, 1]), 1.0, mixed_dataset
    )
    explainer.display_priorities(save_dir=tmp_path, show=False)
    assert len(_pngs(tmp_path)) == 3
