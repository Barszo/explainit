"""Regression counterfactual methods for the standard-methods stage.

Every method exposes the same interface via :class:`BaseRegressionCF`::

    method = SomeMethod(model=..., model_predict=..., X_train=...,
                        feature_names=..., epsilon=...)
    result = method.generate(x, target, bounds, features_to_vary)

where

* ``x``               -- 1D numpy array, the original (scaled) sample,
* ``target``          -- desired (scaled) model output,
* ``bounds``          -- list (len = n_features) of ``(lo, hi)`` or ``None``
                         (``None`` = unbounded for that feature),
* ``features_to_vary`` -- iterable of feature indices allowed to change;
                          every other feature is pinned to ``x``.

``generate`` returns a dict::

    {"cf": np.ndarray | None, "iterations": int, "error": str | None}

Feasible with the currently installed stack (dice-ml, scikit-learn, scipy,
tensorflow); Bayesian optimisation is implemented on scikit-learn's Gaussian
process to avoid an extra hard dependency.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(
    "explainit.experiments.continuous_minlp.standard_methods.methods"
)

Bounds = List[Optional[Tuple[float, float]]]

_BIG = 1e9


class BaseRegressionCF:
    """Common plumbing shared by every regression counterfactual method."""

    name = "base"
    #: whether the method can honour per-feature ``(lo, hi)`` bounds
    supports_bounds = True
    #: whether the method can return more than one distinct counterfactual
    #: (deterministic single-shot methods should set this to ``False``)
    supports_multiple = True

    def __init__(
        self,
        *,
        model: Any = None,
        model_predict: Callable[[np.ndarray], np.ndarray],
        X_train: np.ndarray,
        feature_names: Sequence[str],
        epsilon: float = 0.05,
        **params: Any,
    ) -> None:
        self.model = model
        self.model_predict = model_predict
        self.X_train = np.asarray(X_train, dtype=float)
        self.feature_names = list(feature_names)
        self.epsilon = float(epsilon)
        self.params = dict(params)
        self.feat_min = self.X_train.min(axis=0)
        self.feat_max = self.X_train.max(axis=0)

    # -- helpers ---------------------------------------------------------

    def _predict_one(self, x: np.ndarray) -> float:
        return float(self.model_predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def _resolve_box(
        self,
        x: np.ndarray,
        bounds: Bounds,
        features_to_vary: Iterable[int],
        *,
        finite: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(lo, hi, vary_mask)`` arrays.

        Fixed features get ``lo == hi == x[i]``. Varying features with an
        explicit bound use it; unbounded varying features get either the
        dataset column range (``finite=True``, for samplers) or +/- ``_BIG``
        (``finite=False``, for gradient clipping).
        """
        x = np.asarray(x, dtype=float)
        d = x.shape[0]
        vary = np.zeros(d, dtype=bool)
        for i in features_to_vary:
            vary[int(i)] = True
        lo = np.empty(d)
        hi = np.empty(d)
        for i in range(d):
            if not vary[i]:
                lo[i] = hi[i] = x[i]
                continue
            b = bounds[i] if bounds is not None and i < len(bounds) else None
            if b is None:
                if finite:
                    lo[i], hi[i] = self.feat_min[i], self.feat_max[i]
                else:
                    lo[i], hi[i] = -_BIG, _BIG
            else:
                lo[i], hi[i] = float(b[0]), float(b[1])
            if lo[i] > hi[i]:
                lo[i], hi[i] = hi[i], lo[i]
        return lo, hi, vary

    def _random_start(
        self,
        x: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        vary_idx: np.ndarray,
        rng: np.random.Generator,
        scale: float = 0.2,
    ) -> np.ndarray:
        """Perturb the varying features of ``x`` to seed a diverse restart."""
        z = np.asarray(x, dtype=float).copy()
        if len(vary_idx):
            span = hi[vary_idx] - lo[vary_idx]
            span = np.where(span > 0, span, 1.0)
            z[vary_idx] = np.clip(
                x[vary_idx] + rng.normal(0.0, scale, size=len(vary_idx)) * span,
                lo[vary_idx], hi[vary_idx],
            )
        return z

    def _select_cfs(
        self,
        x: np.ndarray,
        target: float,
        candidates: Sequence[np.ndarray],
        n_cfs: int,
        *,
        fallback: bool = False,
    ) -> List[np.ndarray]:
        """Return up to ``n_cfs`` distinct **valid** counterfactuals, closest
        (by L2) to ``x`` first. If none are valid and ``fallback`` is set, the
        single candidate whose prediction is closest to ``target`` is returned
        (best-effort, so it can still be inspected).
        """
        n_cfs = max(1, int(n_cfs))
        if not len(candidates):
            return []
        arr = np.asarray(candidates, dtype=float).reshape(len(candidates), -1)
        x = np.asarray(x, dtype=float)
        preds = np.asarray(self.model_predict(arr)).reshape(-1)
        errs = np.abs(preds - float(target))
        valid = errs <= self.epsilon
        dists = np.linalg.norm(arr - x, axis=1)
        chosen: List[np.ndarray] = []
        seen: set = set()
        for i in np.argsort(dists):
            if not valid[i]:
                continue
            key = tuple(np.round(arr[i], 6))
            if key in seen:
                continue
            seen.add(key)
            chosen.append(arr[i])
            if len(chosen) >= n_cfs:
                break
        if not chosen and fallback:
            chosen = [arr[int(np.argmin(errs))]]
        return chosen

    def generate_many(
        self,
        x: np.ndarray,
        target: float,
        bounds: Bounds,
        features_to_vary: Iterable[int],
        n_cfs: int = 1,
    ) -> Dict[str, Any]:
        """Generate up to ``n_cfs`` counterfactuals.

        Returns ``{"cfs": List[np.ndarray], "iterations": int | None,
        "error": str | None}``. ``cfs`` may be shorter than ``n_cfs`` (or
        empty) when fewer valid counterfactuals are found.
        """
        raise NotImplementedError

    def generate(
        self,
        x: np.ndarray,
        target: float,
        bounds: Bounds,
        features_to_vary: Iterable[int],
    ) -> Dict[str, Any]:
        out = self.generate_many(x, target, bounds, features_to_vary, 1)
        cfs = out.get("cfs") or []
        return {"cf": cfs[0] if cfs else None,
                "iterations": out.get("iterations"), "error": out.get("error")}


# ---------------------------------------------------------------------------
# Gradient-based family (TensorFlow autodiff through the Keras model)
# ---------------------------------------------------------------------------


class _TFGradientCF(BaseRegressionCF):
    """Shared gradient-descent loop toward ``target`` with a distance penalty.

    Subclasses only define :meth:`_distance_penalty` (a tf scalar computed
    from ``x_var`` and the original ``x_orig``).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.learning_rate = float(self.params.get("learning_rate", 0.05))
        self.max_iterations = int(self.params.get("max_iterations", 1000))
        self.min_iterations = int(self.params.get("min_iterations", 50))
        self.proximity_weight = float(self.params.get("proximity_weight", 0.1))

    def _distance_penalty(self, tf, x_var, x_orig):  # noqa: ANN001
        return tf.reduce_sum(tf.square(x_var - x_orig)) * self.proximity_weight

    def _run(self, x_orig, x_start, target, bounds, features_to_vary):
        """One gradient descent from ``x_start``; proximity is anchored to
        ``x_orig`` and fixed features are pinned to ``x_orig``.
        """
        import tensorflow as tf

        if self.model is None:
            return {"cf": None, "iterations": 0,
                    "error": "gradient method requires a Keras model"}

        x_orig = np.asarray(x_orig, dtype=float)
        x_start = np.asarray(x_start, dtype=float)
        lo, hi, vary = self._resolve_box(x_orig, bounds, features_to_vary, finite=False)
        vary_mask = tf.constant(vary.astype(np.float32))
        lo_t = tf.constant(lo.astype(np.float32))
        hi_t = tf.constant(hi.astype(np.float32))

        anchor = tf.constant(x_orig.astype(np.float32))
        x_var = tf.Variable(x_start.astype(np.float32))
        target_t = tf.constant(float(target), dtype=tf.float32)
        opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        iterations = self.max_iterations
        for it in range(self.max_iterations):
            with tf.GradientTape() as tape:
                pred = self.model(tf.expand_dims(x_var, 0), training=False)[0, 0]
                pred_loss = tf.square(pred - target_t)
                total = pred_loss + self._distance_penalty(tf, x_var, anchor)
            grad = tape.gradient(total, x_var)
            if grad is None:
                iterations = it + 1
                break
            grad = grad * vary_mask
            opt.apply_gradients([(grad, x_var)])
            x_var.assign(tf.clip_by_value(x_var, lo_t, hi_t))
            if it + 1 >= self.min_iterations:
                if abs(float(pred.numpy()) - float(target)) <= self.epsilon:
                    iterations = it + 1
                    break

        cf = x_var.numpy().astype(float)
        cf = np.where(vary, cf, x_orig)  # hard-pin fixed features
        return {"cf": cf, "iterations": int(iterations), "error": None}

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        rng = np.random.default_rng(self.params.get("seed", 0))
        lo, hi, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_idx = np.where(vary)[0]
        n_starts = 1 if n_cfs == 1 else min(n_cfs + 1, 4)
        candidates: List[np.ndarray] = []
        total_iters = 0
        err: Optional[str] = None
        for s in range(n_starts):
            x0 = x if s == 0 else self._random_start(x, lo, hi, vary_idx, rng, scale=0.2)
            out = self._run(x, x0, target, bounds, features_to_vary)
            total_iters += int(out.get("iterations") or 0)
            if err is None:
                err = out.get("error")
            if out.get("cf") is not None:
                candidates.append(out["cf"])
        cfs = self._select_cfs(x, target, candidates, n_cfs, fallback=True)
        return {"cfs": cfs, "iterations": int(total_iters), "error": err}


class WachterRegressionCF(_TFGradientCF):
    """Wachter et al.: prediction loss + L2 proximity penalty."""

    name = "wachter"


class SparseWachterRegressionCF(_TFGradientCF):
    """Wachter with an elastic-net (L2 + L1) penalty for sparser changes."""

    name = "sparse_wachter"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.l1_weight = float(self.params.get("l1_weight", 0.01))

    def _distance_penalty(self, tf, x_var, x_orig):  # noqa: ANN001
        l2 = tf.reduce_sum(tf.square(x_var - x_orig)) * self.proximity_weight
        l1 = tf.reduce_sum(tf.abs(x_var - x_orig)) * self.l1_weight
        return l2 + l1


class PrototypeRegressionCF(_TFGradientCF):
    """Prototype-guided: pulls the CF toward training points whose model
    prediction is already close to ``target`` (regression analogue of the
    class-prototype method).
    """

    name = "prototype"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.n_prototypes = int(self.params.get("n_prototypes", 5))
        self.prototype_weight = float(self.params.get("prototype_weight", 0.1))
        self._pred_train = self.model_predict(self.X_train)

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        import tensorflow as tf

        # Pick the prototypes closest (in predicted target) to the desired value.
        order = np.argsort(np.abs(self._pred_train - float(target)))
        protos = self.X_train[order[: max(1, self.n_prototypes)]]
        self._protos_t = tf.constant(protos.astype(np.float32))
        return super().generate_many(x, target, bounds, features_to_vary, n_cfs)

    def _distance_penalty(self, tf, x_var, x_orig):  # noqa: ANN001
        proximity = tf.reduce_sum(tf.square(x_var - x_orig)) * (self.proximity_weight * 0.1)
        dists = tf.reduce_sum(tf.square(tf.expand_dims(x_var, 0) - self._protos_t), axis=1)
        prototype = tf.reduce_min(dists) * self.prototype_weight
        return proximity + prototype


# ---------------------------------------------------------------------------
# Model-agnostic search family (only needs ``model_predict``)
# ---------------------------------------------------------------------------


class RandomSearchRegressionCF(BaseRegressionCF):
    """Uniform random sampling within the (finite) box; keeps the valid CF
    closest to the original by L2 distance.
    """

    name = "random_search"

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        max_iter = int(self.params.get("max_iterations", 5000))
        seed = self.params.get("seed", None)
        rng = np.random.default_rng(seed)
        lo, hi, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_idx = np.where(vary)[0]

        candidates: List[np.ndarray] = []
        for _ in range(max_iter):
            cand = x.copy()
            cand[vary_idx] = rng.uniform(lo[vary_idx], hi[vary_idx])
            if abs(self._predict_one(cand) - float(target)) <= self.epsilon:
                candidates.append(cand.copy())
        cfs = self._select_cfs(x, target, candidates, n_cfs, fallback=False)
        return {"cfs": cfs, "iterations": int(max_iter), "error": None}


class GrowingSpheresRegressionCF(BaseRegressionCF):
    """Growing Spheres: expand an L2 shell around the sample until a valid CF
    is found, then return the closest one within that shell.
    """

    name = "growing_spheres"

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        n_per_shell = int(self.params.get("n_per_shell", 200))
        max_shells = int(self.params.get("max_shells", 50))
        step = float(self.params.get("step", 0.25))
        seed = self.params.get("seed", None)
        rng = np.random.default_rng(seed)
        lo, hi, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_idx = np.where(vary)[0]
        k = len(vary_idx)
        if k == 0:
            return {"cfs": [], "iterations": 0, "error": "no varying features"}

        iterations = 0
        candidates: List[np.ndarray] = []
        for shell in range(max_shells):
            r_lo = shell * step
            r_hi = (shell + 1) * step
            for _ in range(n_per_shell):
                iterations += 1
                direction = rng.normal(size=k)
                direction /= (np.linalg.norm(direction) + 1e-12)
                radius = rng.uniform(r_lo, r_hi)
                cand = x.copy()
                cand[vary_idx] = np.clip(
                    x[vary_idx] + direction * radius, lo[vary_idx], hi[vary_idx]
                )
                if abs(self._predict_one(cand) - float(target)) <= self.epsilon:
                    candidates.append(cand.copy())
            # Stop expanding once enough valid CFs are found in the closest shells.
            if len(candidates) >= n_cfs:
                break
        cfs = self._select_cfs(x, target, candidates, n_cfs, fallback=False)
        return {"cfs": cfs, "iterations": int(iterations), "error": None}


class NelderMeadRegressionCF(BaseRegressionCF):
    """Gradient-free local search (scipy Nelder-Mead) over the varying
    features, minimising ``(pred - target)^2 + w * L1(change)``.
    """

    name = "nelder_mead"

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        from scipy.optimize import minimize

        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        w = float(self.params.get("proximity_weight", 0.1))
        max_iter = int(self.params.get("max_iterations", 1000))
        rng = np.random.default_rng(self.params.get("seed", 0))
        lo, hi, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_idx = np.where(vary)[0]
        if len(vary_idx) == 0:
            return {"cfs": [], "iterations": 0, "error": "no varying features"}

        def _assemble(z: np.ndarray) -> np.ndarray:
            cand = x.copy()
            cand[vary_idx] = np.clip(z, lo[vary_idx], hi[vary_idx])
            return cand

        def _obj(z: np.ndarray) -> float:
            cand = _assemble(z)
            pred = self._predict_one(cand)
            return (pred - float(target)) ** 2 + w * float(np.sum(np.abs(cand - x)))

        n_starts = 1 if n_cfs == 1 else min(n_cfs + 1, 4)
        candidates: List[np.ndarray] = []
        total_iters = 0
        for s in range(n_starts):
            z0 = x[vary_idx].copy() if s == 0 else \
                self._random_start(x, lo, hi, vary_idx, rng, scale=0.2)[vary_idx]
            res = minimize(
                _obj, z0, method="Nelder-Mead",
                options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-6},
            )
            total_iters += int(getattr(res, "nit", 0) or 0)
            candidates.append(_assemble(np.asarray(res.x, dtype=float)))
        cfs = self._select_cfs(x, target, candidates, n_cfs, fallback=True)
        return {"cfs": cfs, "iterations": int(total_iters), "error": None}


class BayesianOptRegressionCF(BaseRegressionCF):
    """Bayesian optimisation over the varying features using a scikit-learn
    Gaussian process surrogate with Expected-Improvement acquisition.

    Objective minimised: ``|pred - target|``. A small proximity term breaks
    ties toward the original sample.
    """

    name = "bayesian_optimization"

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        from scipy.stats import norm

        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        n_init = int(self.params.get("n_init", 10))
        n_iter = int(self.params.get("n_iter", 40))
        w = float(self.params.get("proximity_weight", 0.05))
        seed = self.params.get("seed", None)
        rng = np.random.default_rng(seed)
        lo, hi, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_idx = np.where(vary)[0]
        k = len(vary_idx)
        if k == 0:
            return {"cfs": [], "iterations": 0, "error": "no varying features"}

        lo_v, hi_v = lo[vary_idx], hi[vary_idx]
        span = np.where(hi_v > lo_v, hi_v - lo_v, 1.0)

        def _assemble(z: np.ndarray) -> np.ndarray:
            cand = x.copy()
            cand[vary_idx] = np.clip(z, lo_v, hi_v)
            return cand

        def _cost(z: np.ndarray) -> float:
            cand = _assemble(z)
            pred = self._predict_one(cand)
            return abs(pred - float(target)) + w * float(np.linalg.norm(cand - x))

        Z = rng.uniform(lo_v, hi_v, size=(n_init, k))
        y = np.array([_cost(z) for z in Z])

        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(k), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)

        for _ in range(n_iter):
            gp.fit(Z, y)
            cand_pool = rng.uniform(lo_v, hi_v, size=(512, k))
            mu, sigma = gp.predict(cand_pool, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            best = y.min()
            improve = best - mu
            zscore = improve / sigma
            ei = improve * norm.cdf(zscore) + sigma * norm.pdf(zscore)
            z_next = cand_pool[int(np.argmax(ei))]
            Z = np.vstack([Z, z_next])
            y = np.append(y, _cost(z_next))

        candidates = [_assemble(z) for z in Z]
        cfs = self._select_cfs(x, target, candidates, n_cfs, fallback=True)
        return {"cfs": cfs, "iterations": int(n_init + n_iter), "error": None}


# ---------------------------------------------------------------------------
# Official DiCE library (regression mode)
# ---------------------------------------------------------------------------


class _SklearnLikeRegressor:
    """Minimal sklearn-style regressor wrapper around a ``predict`` callable,
    so DiCE's model-agnostic (genetic/random) regression search can be used
    with any black-box predictor.
    """

    _estimator_type = "regressor"

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._predict_fn = predict_fn

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._predict_fn(np.asarray(X, dtype=float))).reshape(-1)


class DiceRegressionCF(BaseRegressionCF):
    """Official ``dice-ml`` counterfactuals in regression mode.

    Uses a model-agnostic search (default ``genetic``) with
    ``model_type='regressor'`` and ``desired_range = [target +/- epsilon]``.
    Honours immutability via ``features_to_vary`` and bounds via
    ``permitted_range``.
    """

    name = "dice"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import dice_ml
        import pandas as pd

        self._dice_ml = dice_ml
        self._pd = pd
        self.total_cfs = int(self.params.get("total_cfs", 5))
        self.dice_method = str(self.params.get("method", "genetic"))
        self.backend = str(self.params.get("backend", "sklearn"))

        y_train = self.params.get("y_train")
        if y_train is None:
            y_train = self.model_predict(self.X_train)
        df = pd.DataFrame(self.X_train, columns=self.feature_names)
        df["outcome"] = np.asarray(y_train, dtype=float)

        self._data = dice_ml.Data(
            dataframe=df,
            continuous_features=list(self.feature_names),
            outcome_name="outcome",
        )
        # DiCE's TF2 backend does not support the model-agnostic search methods
        # (genetic/random) for regression, so wrap the predictor in a small
        # sklearn-style object and use the ``sklearn`` backend.
        if self.backend == "sklearn":
            model_obj: Any = _SklearnLikeRegressor(self.model_predict)
        else:
            model_obj = self.model
        self._model = dice_ml.Model(
            model=model_obj, backend=self.backend, model_type="regressor",
        )
        self._exp = dice_ml.Dice(self._data, self._model, method=self.dice_method)

    def generate_many(self, x, target, bounds, features_to_vary, n_cfs=1):
        x = np.asarray(x, dtype=float)
        n_cfs = max(1, int(n_cfs))
        _, _, vary = self._resolve_box(x, bounds, features_to_vary, finite=True)
        vary_names = [self.feature_names[i] for i in np.where(vary)[0]]

        permitted_range: Dict[str, List[float]] = {}
        if bounds is not None:
            for i, name in enumerate(self.feature_names):
                if vary[i] and i < len(bounds) and bounds[i] is not None:
                    permitted_range[name] = [float(bounds[i][0]), float(bounds[i][1])]

        query = self._pd.DataFrame([x], columns=self.feature_names)
        low = float(target) - self.epsilon
        high = float(target) + self.epsilon
        # Oversample so we can keep ``n_cfs`` distinct valid counterfactuals.
        total = max(n_cfs, int(self.total_cfs), n_cfs + 3)
        try:
            res = self._exp.generate_counterfactuals(
                query,
                total_CFs=total,
                desired_range=[low, high],
                features_to_vary=vary_names,
                permitted_range=permitted_range or None,
                verbose=False,
            )
            cf_df = res.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                return {"cfs": [], "iterations": None, "error": None}
            cfs = cf_df[self.feature_names].to_numpy(dtype=float)
            cfs = np.where(vary, cfs, x)  # pin non-varying features (defensive)
            chosen = self._select_cfs(x, target, list(cfs), n_cfs, fallback=False)
            return {"cfs": chosen, "iterations": None, "error": None}
        except Exception as exc:  # pragma: no cover - library edge cases
            return {"cfs": [], "iterations": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


METHODS: Dict[str, type] = {
    DiceRegressionCF.name: DiceRegressionCF,
    WachterRegressionCF.name: WachterRegressionCF,
    SparseWachterRegressionCF.name: SparseWachterRegressionCF,
    PrototypeRegressionCF.name: PrototypeRegressionCF,
    GrowingSpheresRegressionCF.name: GrowingSpheresRegressionCF,
    NelderMeadRegressionCF.name: NelderMeadRegressionCF,
    BayesianOptRegressionCF.name: BayesianOptRegressionCF,
    RandomSearchRegressionCF.name: RandomSearchRegressionCF,
}


def available_methods() -> List[str]:
    return sorted(METHODS)


def build_method(name: str, **kwargs: Any) -> BaseRegressionCF:
    """Instantiate a registered method by name.

    ``kwargs`` must include ``model_predict``, ``X_train``, ``feature_names``
    and ``epsilon``; ``model`` (Keras) is required by the gradient family and
    DiCE. Any extra keys are forwarded as method params.
    """

    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}'. Available: {available_methods()}")
    return METHODS[name](**kwargs)


__all__ = [
    "BaseRegressionCF",
    "METHODS",
    "available_methods",
    "build_method",
]

