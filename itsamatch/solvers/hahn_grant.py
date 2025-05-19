"""Implements the Hahn-Grant algorithm for solving the Quadratic Assignment Problem."""

from time import time

import torch
import tqdm
from torch import Tensor

from .linear_assignment.hungarian_matching import hungarian_matching
from .qap_solver import QAPSolver


class HahnGrant(QAPSolver):
    """
    Hahn-Grant solver for the Quadratic Assignment Problem (QAP).

    This class implements the Hahn-Grant algorithm, an dual ascent method.

    For more details, refer to:
    "Lower bounds for the quadratic assignment problem based upon a dual formulation"
    """

    factorized: bool = False

    def __init__(
        self,
        max_steps_without_improvement: int = 3,
        verbose: bool = False,
        epsilon: float = 1e-6,
    ):
        """
        Initializes the HahnGrant solver.

        Args:
            max_steps_without_improvement (int, optional): Maximum number of iterations
                without improvement in the lower bound before stopping. Defaults to 3.
            verbose (bool, optional): Whether to print progress information. Defaults to False.
            epsilon (float, optional): Tolerance for floating point comparisons and convergence.
                Defaults to 1e-6.
        """
        super().__init__()
        self.max_steps_without_improvement = max_steps_without_improvement
        self.verbose = verbose
        self.epsilon = epsilon

    def solve(
        self,
        cost: Tensor,
        timelimit: float | None = None,
    ) -> tuple[Tensor, bool, float, float | None]:
        """
        Solve the QAP for the given inputs using the Hahn-Grant algorithm.

        The algorithm iteratively refines a lower bound (superleader) and updates
        the cost matrix. In each iteration, it performs a reduction based on
        the Hungarian algorithm applied to 'leader' subproblems.

        Args:
            cost (Tensor): Cost matrix.
            timelimit (float, optional): Time limit for the solver. Defaults to None.

        Returns:
            permutation (Tensor): The permutation that minimizes the cost.
            optimal (bool): Whether the solution is optimal.
            cost (float): The minimum cost.
            bound (float, optional): The lower bound of the cost.
        """
        self._check_size(cost)
        size = cost.shape[0]

        cost_original = cost.clone()  # Keep original cost for final evaluation

        start_time = time()

        # Define infinity based on tensor dtype for handling infeasible assignments
        inf = (
            torch.inf
            if cost.dtype.is_floating_point
            else torch.iinfo(cost.dtype).max
        )
        isinf = cost == inf  # Mask for infinite cost entries

        steps_without_improvement = 0

        superleader = torch.zeros(1, dtype=cost.dtype, device=cost.device)[
            0
        ]  # Initialize lower bound
        iteration = 0
        while True:
            iteration += 1
            # Construct the 'leaders' matrix from diagonal elements of subproblems
            leaders = torch.tensor(
                [[cost[i, j, i, j] for j in range(size)] for i in range(size)]
            )
            # Solve the linear assignment problem for the leaders matrix
            u, v, _, _ = hungarian_matching(leaders)

            # Check for timelimit
            if timelimit is not None and time() - start_time > timelimit:
                break

            improvement = u.sum() + v.sum()

            # Update the superleader (lower bound)
            superleader += improvement
            if self.verbose:
                print(
                    f"Iteration {iteration:<4}: "
                    f"Lower bound: {superleader.item():.4f}"
                )
            # Reduce the leaders matrix
            leaders -= u[:, None] + v[None, :]

            # Update the main cost matrix based on the reduced leaders
            if cost.is_floating_point():
                cost += leaders[:, :, None, None] / (size - 1)
            else:
                cost += leaders[:, :, None, None] // (size - 1)
                # Distribute remainder for integer costs
                for i in range(size):
                    for j in range(size):
                        cost[i, j, : leaders[i, j] % (size - 1), :] += 1

            # Iterate through elements of the reduced leaders matrix in increasing order
            indices = leaders.view(-1).argsort()
            if self.verbose:
                indices = tqdm.tqdm(indices)
            for index in indices:
                i = index // size
                j = index % size
                # Update subproblems in the cost matrix
                cost[i, j, :, :] += cost[:, :, i, j]
                cost[:, :, i, j] = 0  # Zero out the source of the transfer
                cost[isinf] = inf  # Restore infinite costs

                # Solve LAP for the updated subproblem C_ij
                u, v, _, _ = hungarian_matching(cost[i, j])
                # Update the diagonal element C_ijij (leader for this subproblem)
                cost[i, j, i, j] = u.sum() + v.sum()
                # Reduce the subproblem C_ij
                cost[i, j] -= u[:, None] + v[None, :]
                cost[isinf] = inf  # Restore infinite costs
                # Check for timelimit
                if timelimit is not None and time() - start_time > timelimit:
                    break
            if (
                timelimit is not None and time() - start_time > timelimit
            ):  # Break outer loop if inner loop broke due to time
                break

            # Check for convergence
            if improvement < self.epsilon * superleader.abs():
                steps_without_improvement += 1
            else:
                steps_without_improvement = 0

            if (
                steps_without_improvement > self.max_steps_without_improvement
                or (timelimit is not None and time() - start_time > timelimit)
            ):
                break

        # After iterations, construct final leaders matrix and find assignment
        leaders = torch.tensor(
            [[cost[i, j, i, j] for j in range(size)] for i in range(size)]
        )
        _, _, row_ind, col_ind = hungarian_matching(leaders)
        prediction = col_ind[row_ind.argsort()]  # Derive permutation

        cost = cost_original[torch.arange(size), prediction][
            :, torch.arange(size), prediction
        ].sum()
        # Determine if the solution is considered optimal
        optimal = cost - superleader < max(
            self.epsilon, self.epsilon * cost.abs()
        )

        return prediction, optimal.item(), cost.item(), superleader.item()
