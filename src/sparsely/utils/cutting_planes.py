import numpy as np
import mip

from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Result:

    x: np.ndarray
    objective_value: float
    gap: float
    bounds_log: List[Tuple[float, float]]
    incumbent_log: List[np.ndarray]

    @property
    def n_iter(self) -> int:
        return len(self.bounds_log)


def cutting_planes_optimizer(
        model: mip.Model,
        x: mip.LinExprTensor,
        x0: np.ndarray,
        func: Callable[[np.ndarray], Tuple[float, np.ndarray]],
        max_iter: int = 100,
        convergence_tol: float = 1e-3,
        max_seconds_per_cut: Optional[int] = None
) -> Result:

    # Check parameters
    assert max_iter > 1
    assert convergence_tol > 0
    if max_seconds_per_cut is not None:
        assert max_seconds_per_cut > 0

    # Set objective in epigraph form
    y = model.add_var(lb=-mip.INF)
    model.objective = mip.minimize(y)

    # Initialize progress log
    lb, ub = -np.inf, np.inf
    bounds_log = list()
    incumbent_log = list(x0)

    for _ in range(max_iter):

        # Add new cut
        objective_value, gradient = func(x0)
        model.add_constr(y >= objective_value + mip.xsum(gradient * (x - x0)))

        # Re-optimize model and update incumbent solution
        model.optimize(**(dict(max_seconds=max_seconds_per_cut) if max_seconds_per_cut else dict()))
        x0 = np.array([var.x for var in x])

        # Log search progress
        lb, ub = max(lb, model.objective_value), min(ub, objective_value)
        bounds_log.append((lb, ub))
        incumbent_log.append(x0)

        # Check convergence criterion
        gap = abs((objective_value - model.objective_value) / model.objective_value)
        if gap <= convergence_tol:
            break

    return Result(
        x=x0,
        objective_value=model.objective_value,
        gap=gap,
        bounds_log=bounds_log,
        incumbent_log=incumbent_log
    )
