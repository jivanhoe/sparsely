import logging
from typing import Callable, List, Optional, Tuple

import mip
import numpy as np


class CuttingPlanesOptimizer:

    def __init__(
            self,
            func: Callable[[np.ndarray], Tuple[float, np.ndarray]],
            model: Optional[mip.Model] = None,
            x: Optional[mip.LinExprTensor] = None,
            bounds: Optional[Tuple[float, float]] = None,
            max_iter: int = 100,
            convergence_tol: float = 1e-5,
            max_seconds_per_cut: Optional[int] = None,
            minimize: bool = True,
            verbose: bool = False
    ):
        self.func = func
        self.model = model
        self.x = x
        self.bounds = bounds
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol
        self.max_seconds_per_cut = max_seconds_per_cut
        self.minimize = minimize
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

            if self.verbose:
                logging.info(
                    f"Iter: {str(i + 1)} \t"
                    f"Lower bound: {'{:.2e}'.format(self.lb)} \t"
                    f"Incumbent: {'{:.2e}'.format(self.ub)} \t"
                    f"Gap: {'{:.2f}'.format(self.gap * 100)}%"
                )

            # Check convergence criterion
            if self.gap <= self.convergence_tol:
                logging.info(f"Solve completed - convergence tolerance reached")
                break

        if self.gap > self.convergence_tol:
            logging.info(f"Solve terminated - max iterations reached")

        self._solution = np.array([var.x for var in self.x])

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
        if self.model is None:
            assert self.x is None
        assert self.max_iter > 1
        assert self.convergence_tol > 0
        if self.max_seconds_per_cut is not None:
            assert self.max_seconds_per_cut > 0
        if self.model is None:
            assert self.x is None
            assert self.bounds is not None
        if self.bounds is not None:
            assert len(self.bounds) == 2
            assert self.bounds[0] < self.bounds[1]

    def _initialize_solve(self, x0: np.ndarray) -> mip.Var:

        if self.model is None:
            try:
                self.model = mip.Model()
            except mip.InterfacingError:
                self.model = mip.Model(solver_name="CBC")
            self.model.verbose = 0
            self.x = self.model.add_var_tensor(shape=x0.shape, name="x", lb=self.bounds[0], ub=self.bounds[1])
        else:
            assert self.x.shape == x0.shape

        self._bounds_log = list()
        self._gap_log = list()
        self._solution = None
        self._lb = -np.inf
        self._ub = np.inf
        self._gap = np.inf

        y = self.model.add_var(name="y", lb=-mip.INF)
        self.model.objective = mip.minimize(y)
        return y

    def _make_cut(self, x0: np.ndarray, y: mip.Var) -> np.ndarray:

        # Add new cut
        objective_value, gradient = self.func(x0)
        if not self.minimize:
            objective_value *= -1
            gradient *= -1
        self.model.add_constr(y >= objective_value + mip.xsum(gradient * (self.x - x0)))

        # Re-optimize model and update incumbent solution
        self.model.optimize(**(dict(max_seconds=self.max_seconds_per_cut) if self.max_seconds_per_cut else dict()))

        # Log search progress
        self._lb = max(self.lb, self.model.objective_value)
        self._ub = min(self.ub, objective_value)
        self._gap = min(abs((self.ub - self.lb) / (abs(self.lb) + 1e-10)), self.gap)
        self._bounds_log.append((self.lb, self.ub))
        self._gap_log.append(self._gap)

        return np.array([var.x for var in self.x])
