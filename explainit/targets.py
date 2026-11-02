"""What a counterfactual has to achieve."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    """Either a prediction ``value`` within ``tolerance`` or a predicted class.

    Build it with :func:`resolve_target`.
    """

    value: float | None = None
    tolerance: float | None = None
    target_class: int | None = None
    threshold: float | None = None

    @property
    def is_class(self) -> bool:
        return self.target_class is not None

    def is_reached(self, prediction: float) -> bool:
        """Whether a model prediction reaches the target."""
        if self.is_class:
            assert self.threshold is not None
            if self.target_class == 1:
                return bool(prediction > self.threshold)
            return bool(prediction < self.threshold)
        assert self.value is not None
        assert self.tolerance is not None
        return bool(abs(prediction - self.value) <= self.tolerance)


def resolve_target(
    target: float | None, tolerance: float | None, target_class: int | None, threshold: float
) -> Target:
    """Validate the target arguments of ``explain``.

    Exactly one of ``target`` (a prediction value, reached within
    ``tolerance``) and ``target_class`` (0 or 1, on the corresponding side of
    ``threshold``) must be given.
    """
    if (target is None) == (target_class is None):
        raise ValueError("pass exactly one of `target` (a value) or `target_class` (0 or 1)")
    if target_class is not None:
        if target_class not in (0, 1):
            raise ValueError(f"target_class must be 0 or 1, got {target_class!r}")
        return Target(target_class=int(target_class), threshold=float(threshold))
    if tolerance is None or tolerance < 0:
        raise ValueError(f"tolerance must be a non-negative number, got {tolerance!r}")
    return Target(value=target, tolerance=tolerance)
