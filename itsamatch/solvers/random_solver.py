"""Random solver for the Quadratic Assignment Problem (QAP)."""

import torch
from torch import Tensor

from .qap_solver import QAPSolver


class RandomSolver(QAPSolver):
    """
    Random solver for the Quadratic Assignment Problem (QAP).
    This solver generates a random permutation of the input size and computes the cost.
    """

    factorized: bool = True

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
        self._check_size_factorized(cost1, cost2, linear, constant)
        size = cost1.shape[0]

        prediction = torch.randperm(size, device=cost1.device)

        # Compute the cost
        cost = cost1 * cost2[prediction][:, prediction]

        if linear is not None:
            cost += linear[torch.arange(size, device=cost1.device), prediction]
        if constant is not None:
            cost += constant

        return prediction, False, cost, None
