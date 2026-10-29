"""Best counterfactuals found so far during the MINLP refinement loop."""

from dataclasses import dataclass


@dataclass
class Incumbents:
    """Two incumbents, so that the priority benefit decides the answer.

    * ``feasible``: among candidates inside the target band, the one with the
      highest priority benefit. Returned whenever it exists.
    * ``infeasible``: only used while no candidate has entered the band;
      ranked by distance to the target.
    """

    feasible: list | None = None
    feasible_priority: float = float("-inf")
    feasible_distance: float = float("inf")
    feasible_source: str | None = None
    feasible_prediction: float | None = None
    infeasible: list | None = None
    infeasible_distance: float = float("inf")
    infeasible_prediction: float | None = None

    @property
    def reached_target(self) -> bool:
        return self.feasible is not None

    def set_anchor(self, cf, priority, distance, prediction=None) -> None:
        """Start from an anchor that is already inside the band."""
        self.feasible = cf
        self.feasible_prediction = prediction
        self.feasible_priority = priority if priority is not None else float("-inf")
        self.feasible_distance = distance
        self.feasible_source = "anchor"

    def offer(self, cf, evaluation) -> bool:
        """Consider a candidate; returns whether it became an incumbent."""
        priority = evaluation["priority"]
        if evaluation["feasible"]:
            if self.feasible is None or (
                priority is not None and priority > self.feasible_priority
            ):
                self.feasible = list(cf)
                self.feasible_priority = priority if priority is not None else float("-inf")
                self.feasible_distance = evaluation["distance"]
                self.feasible_prediction = evaluation.get("model_pred")
                self.feasible_source = "optimiser"
                return True
        elif self.feasible is None and evaluation["distance"] < self.infeasible_distance:
            self.infeasible = list(cf)
            self.infeasible_distance = evaluation["distance"]
            self.infeasible_prediction = evaluation.get("model_pred")
            return True
        return False

    def describe(self) -> tuple[str, str]:
        """``(best priority, best distance)`` formatted for the log."""
        priority = (
            f"{self.feasible_priority:.4f}" if self.feasible_priority != float("-inf") else "n/a"
        )
        if self.feasible is not None:
            distance = f"{self.feasible_distance:.4f}"
        elif self.infeasible_distance != float("inf"):
            distance = f"{self.infeasible_distance:.4f}"
        else:
            distance = "n/a"
        return priority, distance
