from time import time
from warnings import warn

import torch
from scipy.optimize import quadratic_assignment
from torch import Tensor

from ._2opt import _quadratic_assignment_2opt
from .qap_solver import QAPSolver


class ScipySolver(QAPSolver):
    """
    Scipy solver for the Quadratic Assignment Problem (QAP).

    This solver can utilize SciPy's `quadratic_assignment` function, including
    methods like 'faq' (Fast Approximate QAP), or a custom '2opt' implementation.
    It supports factorized cost matrices.
    """

    factorized: bool = True

    def __init__(
        self,
        method: str = "2opt",
        num_repetitions: int = 100,
        transpose_first: bool = False,
    ):
        """
        Initialize the ScipySolver.

        Args:
            method (str, optional): The optimization method to use.
                Can be "2opt" for a custom 2-opt algorithm, or any method
                supported by `scipy.optimize.quadratic_assignment` (e.g., "faq").
                Defaults to "2opt".
            num_repetitions (int, optional): The number of times to run the
                optimization algorithm, each with a new random seed. The best
                result over these repetitions is returned. Defaults to 100.
            transpose_first (bool, optional): If True, the first cost matrix
                (cost1) will be transposed before solving. This may affect
                the problem formulation for certain QAP variants.
                Defaults to False.
        """
        super().__init__()
        self.method = method
        self.num_repetitions = num_repetitions
        self.transpose_first = transpose_first

    def solve(
        self,
        cost1: Tensor,
        cost2: Tensor,
        linear: Tensor = None,
        constant: Tensor = None,
        timelimit: float | None = None,
    ) -> tuple[Tensor, bool, float, float | None]:
        """
        Solve the QAP for the given inputs.

        The objective function is to minimize:
        `trace(cost1 @ P @ cost2.T @ P.T) + trace(linear @ P.T) + constant`
        where P is a permutation matrix.

        Args:
            cost1 (Tensor): First factorized cost matrix.
            cost2 (Tensor): Second factorized cost matrix.
            linear (Tensor, optional): Linear term. Defaults to None.
            constant (Tensor, optional): Constant term. Defaults to None.
            timelimit (float, optional): Time limit for the solver. Defaults to None.

        Returns:
            permutation (Tensor): The permutation that minimizes the cost.
            optimal (bool): Whether the solution is optimal.
            cost (float): The minimum cost.
            bound (float, optional): The lower bound of the cost.
        """
        # Validate input tensor sizes
        self._check_size_factorized(cost1, cost2, linear, constant)

        # Issue warnings for unsupported option combinations
        if linear is not None and self.transpose_first:
            warn("linear is not supported with the transpose_first option. ")

        if linear is not None and self.method == "faq":
            warn("linear is not supported with the faq method. ")

        # Initialize linear term if not provided
        if linear is None:
            linear = torch.zeros_like(cost1)

        # Initialize constant term if not provided
        if constant is None:
            constant = torch.zeros(1, device=cost1.device, dtype=cost1.dtype)

        # Set default timelimit if not provided
        if timelimit is None:
            timelimit = float("inf")

        # Optionally transpose the first cost matrix
        # (used for LocalCKA (https://arxiv.org/pdf/2401.05224))
        if self.transpose_first:
            cost1 = cost1.T

        # Perform optimization based on the selected method
        if self.method == "2opt":
            start_time = time()
            best_objective = torch.inf  # Initialize for minimization
            best_permutation = None

            # Run 2-opt multiple times with different seeds
            for _ in range(self.num_repetitions):
                subseed = torch.randint(
                    0, 2**32 - 1, (1,), dtype=torch.long
                ).item()

                # Call the custom 2-opt solver
                res = _quadratic_assignment_2opt(
                    cost1.cpu().numpy(),
                    cost2.cpu().numpy(),
                    linear.cpu().numpy(),
                    constant.cpu().numpy(),
                    maximize=False,
                    rng=subseed,
                )

                # Update if a better solution (lower cost) is found
                if res.fun < best_objective:
                    best_objective = res.fun
                    best_permutation = res.col_ind

                # Check for timelimit
                if time() - start_time > timelimit:
                    break

            # Convert the best permutation to a tensor
            prediction = torch.as_tensor(
                best_permutation, device=cost1.device, dtype=torch.long
            )
        else:  # For SciPy's built-in methods (currently only "faq")
            start_time = time()
            best_objective = torch.inf  # Initialize for minimization
            best_permutation = None

            # Run SciPy solver multiple times with different seeds
            for _ in range(self.num_repetitions):
                subseed = torch.randint(
                    0, 2**32 - 1, (1,), dtype=torch.long
                ).item()

                # Call SciPy's quadratic_assignment solver
                # Note: The 'linear' tensor is not explicitly passed to this call.
                # SciPy's 'costs' parameter could handle linear terms, but isn't used here.
                res = quadratic_assignment(
                    cost1.cpu().numpy(),
                    cost2.cpu().numpy(),
                    method=self.method,
                    options={"maximize": False, "rng": subseed},
                )

                # Update if a "better" solution is found (higher res.fun as per current logic)
                if res.fun < best_objective:
                    best_objective = res.fun
                    best_permutation = res.col_ind

                # Check for timelimit
                if time() - start_time > timelimit:
                    break

            # Convert the best permutation to a tensor
            prediction = torch.as_tensor(
                best_permutation, device=cost1.device, dtype=torch.long
            )

            # Add the constant term to the objective value
            # This is done outside the loop, applied to the best_objective found.
            # For "2opt", the constant is handled within _quadratic_assignment_2opt.
            best_objective += constant.item()

        return prediction, False, best_objective, None
