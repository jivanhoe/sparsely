import logging
from typing import Callable, List, Optional, Tuple, Union

import mip
import numpy as np


class CuttingPlanesOptimizer:

    def __init__(
            self,
            func: Callable[[np.ndarray], Tuple[float, np.ndarray]],
            model: Optional[mip.Model] = None,
            x: Optional[mip.LinExprTensor] = None,
            bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
            max_iter: int = 100,
            convergence_tol: float = 1e-6,
            max_seconds_per_cut: Optional[int] = None,
            verbose: bool = False
    ):
        self.func = func
        self.model = model
        self.x = x
        self.bounds = bounds
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.max_seconds_per_cut = max_seconds_per_cut
        self.verbose = verbose

    def optimize(self, x0: np.ndarray) -> None:

        self._check_parameters()
        y = self._initialize_solve(x0=x0)

        if self.verbose:
            logging.info(
                f"Commencing cutting planes solve on model with {self.model.num_cols - 1} variable(s) "
                f"({self.model.num_int} integer) and {self.model.num_rows} constraint(s)"
            )

        for i in range(self.max_iter):

            x0 = self._make_cut(x0=x0, y=y)

            if x0 is None:
                logging.info(f"Solve terminated - no solution found. Solver status: {self.model.status.name}")
                break

            if self.verbose:
                logging.info(
                    f"Iter: {str(i + 1)} \t "
                    f"| Current: {'{:.2e}'.format(self._current)}\t"
                    f"| Best: {'{:.2e}'.format(self.ub)}\t"
                    f"| Bound: {'{:.2e}'.format(self.lb)}\t"
                    f"| Gap: {'{:.2f}'.format(self.gap * 100)}%"
                )

            # Check convergence criterion
            if self.gap <= self.convergence_tol:
                logging.info(f"Solve completed - convergence tolerance reached")
                break

        if self.model.status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            if self.gap > self.convergence_tol:
                logging.info(f"Solve terminated - max iterations reached")

    @property
    def lb(self) -> float:
        return self._lb

    @property
    def ub(self) -> float:
        return self._ub

    @property
    def gap(self) -> float:
        return self._gap

    @property
    def objective_value(self) -> float:
        return self.model.objective_value

    @property
    def solution(self) -> np.ndarray:
        return self._solution

    @property
    def bounds_log(self) -> List[Tuple[float, float]]:
        return self._bounds_log

    @property
    def gap_log(self) -> List[float]:
        return self._gap_log

    @property
    def n_iter(self) -> int:
        return len(self.bounds_log)

    def _check_parameters(self) -> None:
        if self.max_iter <= 1 and isinstance(self.max_iter, int):
            raise ValueError(f"Max iterations must be an integer greater than 1 - provided value is: {self.max_iter}.")
        if self.convergence_tol <= 0:
            raise ValueError(f"Convergence tolerance must be positive - provided value is: {self.max_seconds_per_cut}.")
        if self.max_seconds_per_cut is not None:
            if self.max_seconds_per_cut <= 0:
                raise ValueError(
                    f"Max seconds per cut must be positive - provided value is: {self.max_seconds_per_cut}."
                )
        if self.model is None:
            if self.x is not None:
                raise ValueError("If a model is provided, the decision variable 'x' cannot be provided.")
            if self.bounds is None:
                raise ValueError(
                    "If no model is provided, variable bounds must be specified - None provided."
                )
        if self.bounds:
            if isinstance(self.bounds[0], (float, int)):
                if len(self.bounds) != 2:
                    raise ValueError
                if self.bounds[0] >= self.bounds[1]:
                    raise ValueError
                if not all(isinstance(x, (float, int)) for x in self.bounds):
                    raise ValueError
            else:
                if len(self.bounds) < 1:
                    raise ValueError
                if any(len(x) != 2 for x in self.bounds):
                    raise ValueError
                if not all(isinstance(x, (float, int)) for bound in self.bounds for x in bound):
                    raise ValueError

    def _initialize_solve(self, x0: np.ndarray) -> mip.Var:

        if self.model is None:

            # Set up model
            try:
                self.model = mip.Model()
            except mip.InterfacingError:
                self.model = mip.Model(solver_name="CBC")
            self.model.verbose = 0

            # Add decision variables with bounds
            if isinstance(self.bounds[0], (float, int)):
                self.x = self.model.add_var_tensor(shape=x0.shape, name="x", lb=self.bounds[0], ub=self.bounds[1])
            else:
                if len(self.bounds) != len(x0):
                    raise ValueError
                self.x = self.model.add_var_tensor(shape=x0.shape, name="x")
                for i, (lb, ub) in self.bounds:
                    self.model.add_constr(self.x[i] >= lb)
                    self.model.add_constr(self.x[i] <= ub)

        else:
            if self.x.shape != x0.shape:
                raise ValueError

        self._bounds_log = list()
        self._gap_log = list()
        self._solution = None
        self._lb = -np.inf
        self._ub = np.inf
        self._current = np.inf
        self._gap = np.inf

        y = self.model.add_var(name="y", lb=-mip.INF)
        self.model.objective = mip.minimize(y)
        return y

    def _make_cut(self, x0: np.ndarray, y: mip.Var) -> Optional[np.ndarray]:

        # Add new cut
        objective_value, gradient = self.func(x0)
        self.model.add_constr(y >= objective_value + mip.xsum(gradient * (self.x - x0)))

        # Re-optimize model and update incumbent solution
        self.model.optimize(**(dict(max_seconds=self.max_seconds_per_cut) if self.max_seconds_per_cut else dict()))

        if self.model.status in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):

            # Update search progress
            x0 = np.array([var.x for var in self.x])
            self._lb = max(self.lb, self.model.objective_value)
            self._current = objective_value
            if self._current < self.ub:
                self._ub = objective_value
                self._solution = x0
            self._gap = min(abs((self.ub - self.lb) / (abs(self.ub) + 1e-10)), self.gap)
            self._bounds_log.append((self.lb, self.ub))
            self._gap_log.append(self._gap)

            return x0
