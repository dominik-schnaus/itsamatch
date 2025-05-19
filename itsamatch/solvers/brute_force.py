"""Module for the brute force QAP solver."""

from itertools import permutations
from warnings import warn

import torch
from torch import Tensor

from .qap_solver import QAPSolver


class BruteForce(QAPSolver):
    """
    Brute force solver for the Quadratic Assignment Problem (QAP).
    This solver computes all possible permutations and selects the one with the minimum cost.
    """

    factorized: bool = True

    # Cache for storing permutations to avoid recomputing them
    permutation_cache: dict = {}

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
            timelimit (float, optional): Time limit for the solver. Defaults to None. Not supported in brute force.

        Returns:
            permutation (Tensor): The permutation that minimizes the cost.
            optimal (bool): Whether the solution is optimal.
            cost (float): The minimum cost.
            bound (float, optional): The lower bound of the cost.
        """
        # Check if the input tensors have compatible sizes
        self._check_size_factorized(cost1, cost2, linear, constant)
        size = cost1.shape[0]

        if timelimit is not None:
            warn(
                "timelimit is not supported in the brute force solver. "
                "The solver will run until all permutations are computed."
            )

        # Check if the permutation has already been computed and cached
        if size in self.permutation_cache:
            perms = self.permutation_cache[size].to(device=cost1.device)
        else:
            # Generate all permutations of a given size
            perms = torch.as_tensor(
                list(permutations(range(size))), device=cost1.device
            )

            # Cache the permutations on CPU to save GPU memory
            self.permutation_cache[size] = perms.cpu()

        # Compute the cost for each permutation
        # Permute the second cost matrix according to each permutation
        # perms[:, None, :] creates a view for broadcasting: (num_perms, 1, size)
        # .repeat(1, size, 1) expands it to (num_perms, size, size) for gathering
        # cost2[perms] permutes rows of cost2: (num_perms, size, M)
        # torch.gather permutes columns based on perms for each permutation
        cost2_permuted = torch.gather(
            cost2[perms], 2, perms[:, None, :].repeat(1, size, 1)
        )

        # Calculate the quadratic term of the cost
        cost = (cost1[None] * cost2_permuted).sum(dim=(1, 2))

        # Add the linear term to the cost if provided
        if linear is not None:
            cost += linear[torch.arange(size), perms].sum(dim=2)

        # Add the constant term to the cost if provided
        if constant is not None:
            cost += constant

        # Find the permutation with the minimum cost
        min_cost, min_index = cost.min(dim=0)
        min_permutation = perms[min_index]

        # Return the best permutation, optimality (always true for brute force), cost, and bound (same as cost)
        return min_permutation, True, min_cost.item(), min_cost.item()
