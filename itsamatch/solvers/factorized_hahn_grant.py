"""
Implements the Factorized Hahn-Grant solver for the Quadratic Assignment Problem (QAP).
This solver is an iterative dual ascent method for solving QAPs. It leverages different
linear assignment problem (LAP) solvers.

For more details, see Alg. 2 of our paper:
"It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data"
(https://arxiv.org/pdf/2503.24129)
"""

from functools import partial
from time import time

import torch
from lapjv import lapjv
from scipy.optimize import quadratic_assignment
from torch import Tensor
from tqdm import tqdm, trange

from ._2opt import _quadratic_assignment_2opt
from .linear_assignment.auction_lap import auction_lap
from .linear_assignment.forward_reverse_auction_cpp import (
    get_forward_backward_auction_cpp,
)
from .linear_assignment.hungarian_matching import hungarian_matching
from .qap_solver import QAPSolver


def lapjv_interface(
    cost_matrix: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Interface for the LAPJV solver.

    Args:
        cost_matrix (Tensor): The cost matrix for the LAP.

    Returns:
        u (Tensor): Dual variables for the rows.
        v (Tensor): Dual variables for the columns.
        row_ind (Tensor): Row indices of the optimal assignment.
        col_ind (Tensor): Column indices of the optimal assignment.
    """
    cost_matrix_numpy = cost_matrix.cpu().numpy()
    col_ind, _, (_, u, v) = lapjv(cost_matrix_numpy, force_doubles=False)
    col_ind = torch.as_tensor(
        col_ind, dtype=torch.long, device=cost_matrix.device
    )
    row_ind = torch.arange(
        cost_matrix.shape[0], dtype=col_ind.dtype, device=cost_matrix.device
    )
    u = torch.as_tensor(u, dtype=cost_matrix.dtype, device=cost_matrix.device)
    v = torch.as_tensor(v, dtype=cost_matrix.dtype, device=cost_matrix.device)
    return u, v, row_ind, col_ind


def update_best_permutation(
    best_cost,
    best_permutation,
    row_ind,
    col_ind,
    source_kernel,
    target_kernel,
    first_order_constant,
    constant,
    eps=1e-8,
) -> tuple[float, Tensor]:
    """
    Updates the best permutation found so far if the current permutation yields a lower cost.

    Args:
        best_cost (float): The current best cost.
        best_permutation (Tensor): The current best permutation.
        row_ind (Tensor): Row indices of the current assignment.
        col_ind (Tensor): Column indices of the current assignment.
        source_kernel (Tensor): The first kernel matrix (A).
        target_kernel (Tensor): The second kernel matrix (B).
        first_order_constant (Tensor): The linear term matrix (L).
        constant (Tensor): The constant term.
        eps (float, optional): Epsilon for floating point comparisons. Defaults to 1e-8.

    Returns:
        cost (float): The cost of the current assignment.
        best_permutation (Tensor): The best permutation found so far.
    """
    cost = (
        (
            source_kernel[row_ind][:, row_ind]
            * target_kernel[col_ind][:, col_ind]
        ).sum()
        + first_order_constant[row_ind, col_ind].sum()
        + constant
    )
    if cost + eps < best_cost:
        return cost, col_ind[row_ind.argsort()]
    return best_cost, best_permutation


class FactorizedHahnGrant(QAPSolver):
    """
    Factorized Hahn-Grant solver for the Quadratic Assignment Problem (QAP).

    This solver uses a dual ascent method to iteratively improve the solution.
    It leverages different linear assignment problem (LAP) solvers to find the optimal
    assignment between two factorized cost matrices.
    The objective is to minimize
    :math:`tr(cost1 P cost2^T P^T) + tr(linear P^T) + constant`,
    where P is the permutation matrix.
    """

    factorized: bool = True

    # Dictionary indicating if a matching algorithm requires epsilon scaling.
    needs_epsilon_dict = {
        "auction": True,
        "hungarian": False,
        "lapjv": False,
        "forward_backward_auction_cpp": True,
    }

    # Dictionary mapping matching algorithm names to their respective functions.
    matching_algorithms = {
        "auction": partial(auction_lap, minimize=True),
        "hungarian": hungarian_matching,
        "lapjv": lapjv_interface,
    }

    def _get_matching_algorithm(self, matching_algorithm: str, **kwargs):
        """
        Returns the specified matching algorithm function.

        Args:
            matching_algorithm (str): Name of the matching algorithm.

        Returns:
            Callable: The matching algorithm function.
        """
        if matching_algorithm == "forward_backward_auction_cpp":
            return get_forward_backward_auction_cpp(minimize=True)
        return self.matching_algorithms[matching_algorithm]

    def __init__(
        self,
        matching_algorithm: str = "lapjv",
        max_steps_without_improvement: int = 3,
        verbose: bool = False,
        epsilon: float = 1e-6,
        initialize_with_2opt_faq: bool = True,
        num_repetitions_2opt_faq: int = 100,
        update_with_lap_solutions: bool = True,
        epsilon_auction: float | None = 0.1,
        decay_auction: float | None = 0.9,
    ):
        """
        Initializes the FactorizedHahnGrant solver.

        Args:
            matching_algorithm (str, optional): LAP solver to use.
                Options: "lapjv", "hungarian", "auction", "forward_backward_auction_cpp".
                Defaults to "lapjv".
            max_steps_without_improvement (int, optional): Maximum number of iterations
                without improvement in the lower bound before stopping. Defaults to 3.
            verbose (bool, optional): Whether to print progress information. Defaults to False.
            epsilon (float, optional): Tolerance for floating point comparisons and convergence.
                Defaults to 1e-6.
            initialize_with_2opt_faq (bool, optional): Whether to initialize the best solution
                using FAQ and 2-opt heuristics. Defaults to True.
            num_repetitions_2opt_faq (int, optional): Number of repetitions for FAQ/2-opt
                initialization. Defaults to 100.
            update_with_lap_solutions (bool, optional): Whether to update the best solution
                with solutions from intermediate LAPs. Defaults to True.
            epsilon_auction (float | None, optional): Epsilon parameter for auction-based
                LAP solvers. Defaults to 0.1.
            decay_auction (float | None, optional): Decay rate for epsilon_auction.
                Defaults to 0.9.
        """
        super().__init__()

        self.matching_algorithm_str = matching_algorithm

        self.matching_algorithm = self._get_matching_algorithm(
            matching_algorithm
        )
        self.max_steps_without_improvement = max_steps_without_improvement
        self.verbose = verbose
        self.epsilon = epsilon
        self.initialize_with_2opt_faq = initialize_with_2opt_faq
        self.num_repetitions_2opt_faq = num_repetitions_2opt_faq
        self.update_with_lap_solutions = update_with_lap_solutions

        self.epsilon_auction = epsilon_auction
        self.decay_auction = decay_auction
        self.needs_epsilon = self.needs_epsilon_dict.get(
            matching_algorithm, False
        )
        if self.needs_epsilon and (
            self.epsilon_auction is None or self.decay_auction is None
        ):
            raise ValueError(
                f"Matching algorithm {self.matching_algorithm} requires epsilon and decay_auction, but it is not provided."
            )

    def solve(
        self,
        cost1: Tensor,
        cost2: Tensor,
        linear: Tensor = None,
        constant: Tensor = None,
        timelimit: float | None = None,
    ) -> tuple[Tensor, bool, float, float | None]:
        """
        Solve the QAP for the given factorized costs and linear term.
        The objective is to minimize tr(cost1 P cost2 P^T) + tr(linear P^T) + constant.

        Args:
            cost1 (Tensor): First factorized cost matrix (A).
            cost2 (Tensor): Second factorized cost matrix (B).
            linear (Tensor, optional): Linear term matrix (L). Defaults to None (zeros).
            constant (Tensor, optional): Constant term. Defaults to None (zero).
            timelimit (float, optional): Time limit for the solver in seconds. Defaults to None (no limit).

        Returns:
            permutation (Tensor): The permutation matrix P (as column indices) that minimizes the cost.
            optimal (bool): Whether the solution is proven optimal.
            cost (float): The minimum cost found.
            bound (float, optional): The lower bound of the cost.
        """
        self._check_size_factorized(cost1, cost2, linear, constant)

        if linear is None:
            linear = torch.zeros_like(cost1)

        if constant is None:
            constant = torch.zeros(1, dtype=cost1.dtype, device=cost1.device)

        if timelimit is None:
            timelimit = float("inf")

        n = cost1.shape[0]
        cost1_min = cost1.min()
        cost1_sum = cost1.sum()
        cost2_min = cost2.min()
        cost2_sum = cost2.sum()

        cost1 -= cost1_min
        cost2 -= cost2_min
        constant -= (
            -cost2_min * cost1_sum
            - cost1_min * cost2_sum
            + cost1_min * cost2_min * n**2
        )

        leaders = torch.outer(torch.diag(cost1), torch.diag(cost2)) + linear

        start_time = time()

        is_floating_point = (
            cost1.dtype.is_floating_point or cost2.dtype.is_floating_point
        )
        eps = self.epsilon if is_floating_point else 0
        eps_auction = self.epsilon_auction if is_floating_point else 0

        if is_floating_point:
            if cost1.dtype.is_floating_point:
                cost2 = cost2.to(cost1)
            elif cost2.dtype.is_floating_point:
                cost1 = cost1.to(cost2)
        else:
            cost2 = cost2.to(cost1)

        superleader = constant.clone()
        us = torch.zeros(n, n, n - 1, dtype=cost1.dtype, device=cost1.device)
        vs = torch.zeros(n, n, n - 1, dtype=cost1.dtype, device=cost1.device)

        best_permutation = None
        best_cost = (
            torch.inf * torch.ones(1, dtype=cost1.dtype, device=cost1.device)[0]
        )

        steps_without_improvement = 0

        optimal = False

        iteration = 0
        while True:
            iteration += 1
            # solver the linear term
            if self.needs_epsilon:
                u, v, row_ind, col_ind = self.matching_algorithm(
                    leaders, eps=eps_auction
                )
            else:
                u, v, row_ind, col_ind = self.matching_algorithm(leaders)

            # update if the LAP solution is better
            if self.update_with_lap_solutions:
                best_cost, best_permutation = update_best_permutation(
                    best_cost,
                    best_permutation,
                    row_ind,
                    col_ind,
                    cost1,
                    cost2,
                    linear,
                    constant,
                )

                if time() - start_time > timelimit:
                    break

            # Use primal heuristics to initialize the best solution
            if iteration == 1 and self.initialize_with_2opt_faq:
                if self.verbose:
                    r = trange(self.num_repetitions_2opt_faq)
                else:
                    r = range(self.num_repetitions_2opt_faq)

                cost1_numpy = cost1.cpu().numpy()
                cost2_numpy = cost2.cpu().numpy()

                # Use multiple random seeds to initialize the best solution
                for i in r:
                    if i == 0 and best_permutation is not None:
                        partial_guess = (
                            torch.stack(
                                [
                                    torch.arange(
                                        n, dtype=torch.long, device=cost1.device
                                    ),
                                    best_permutation,
                                ],
                                dim=1,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        partial_guess = None
                    subseed = torch.randint(
                        0, 2**32 - 1, (1,), dtype=torch.long
                    ).item()

                    # Use the FAQ solver
                    res = quadratic_assignment(
                        cost1_numpy,
                        cost2_numpy,
                        method="faq",
                        options={
                            "maximize": False,
                            "rng": subseed,
                        },
                    )
                    row_ind = torch.arange(
                        n, dtype=torch.long, device=cost1.device
                    )
                    col_ind = torch.as_tensor(
                        res.col_ind, dtype=torch.long, device=cost1.device
                    )
                    best_cost, best_permutation = update_best_permutation(
                        best_cost,
                        best_permutation,
                        row_ind,
                        col_ind,
                        cost1,
                        cost2,
                        linear,
                        constant,
                    )

                    if time() - start_time > timelimit:
                        break

                    # Use the 2-opt solver
                    res = _quadratic_assignment_2opt(
                        cost1.cpu().numpy(),
                        cost2.cpu().numpy(),
                        linear.cpu().numpy(),
                        constant.cpu().numpy(),
                        maximize=False,
                        rng=subseed,
                        partial_guess=partial_guess,
                    )
                    if res.fun < best_cost:
                        best_cost = torch.as_tensor(
                            [res.fun], dtype=torch.float, device=cost1.device
                        ).sum()
                        best_permutation = torch.as_tensor(
                            res.col_ind, dtype=torch.long, device=cost1.device
                        )
                    if self.verbose:
                        print(
                            f"Iteration {iteration:<4}.{i}: "
                            f"Best cost: {best_cost.item():.4f}, "
                            f"Lower bound: {superleader.item():.4f}, "
                            f"Duality gap: {best_cost.item() - superleader.item():.4f}, "
                            f"Relative duality gap: {(best_cost.item() - superleader.item()) / best_cost.abs().item():.4f}",
                            f"Best permutation: {best_permutation}",
                        )
                    if time() - start_time > timelimit:
                        break

            if time() - start_time > timelimit:
                break

            # update the superleader (lower bound) with the gains from the LAP
            improvement = u.sum() + v.sum()
            superleader += improvement

            if self.verbose:
                print(
                    f"Iteration {iteration:<4}: "
                    f"Best cost: {best_cost.item():.4f}, "
                    f"Lower bound: {superleader.item():.4f}, "
                    f"Relative improvement: {improvement.item() / best_cost.abs().item():.6f}",
                    f"Duality gap: {best_cost.item() - superleader.item():.4f}, "
                    f"Relative duality gap: {(best_cost.item() - superleader.item()) / best_cost.abs().item():.4f}",
                    f"Best permutation: {best_permutation}",
                )

            leaders -= u[:, None] + v[None, :]

            if iteration > 1 and best_cost - superleader < max(
                eps, eps * best_cost.abs()
            ):
                optimal = True
                break

            # redistribute the remaining leader costs
            if is_floating_point:
                us -= leaders[:, :, None] / (n - 1)
            else:
                us -= leaders[:, :, None] // (n - 1)
                for i in range(n):
                    for j in range(n):
                        us[i, j, : leaders[i, j] % (n - 1)] -= 1

            # first update the indices with the highest possible gains
            indices = leaders.view(-1).argsort()
            if self.verbose:
                indices = tqdm(indices)
            for index in indices:
                i = index // n
                j = index % n

                # remove the i-th row and j-th column from the cost matrix
                valid_rows = torch.where(torch.arange(n) != i)[0]
                valid_cols = torch.where(torch.arange(n) != j)[0]

                us_selected = us[valid_rows, :][:, valid_cols]
                vs_selected = vs[valid_rows, :][:, valid_cols]

                arange = torch.arange(n - 1)
                row_selected_i = i - (arange < i).long()
                col_selected_j = j - (arange < j).long()

                cost_matrix = (
                    2 * torch.outer(cost1[valid_rows, i], cost2[valid_cols, j])
                    - us[i, j, :, None]
                    - vs[i, j, None, :]
                    - us_selected[arange, :, row_selected_i]
                    - vs_selected[:, arange, col_selected_j]
                )

                # solve the i, j-th LAP
                if self.needs_epsilon:
                    u, v, row_ind, col_ind = self.matching_algorithm(
                        cost_matrix, eps=eps_auction
                    )
                else:
                    u, v, row_ind, col_ind = self.matching_algorithm(
                        cost_matrix
                    )

                row_ind[row_ind >= i] += 1
                col_ind[col_ind >= j] += 1
                row_ind_full = torch.cat(
                    [
                        row_ind,
                        torch.tensor(
                            [i], dtype=row_ind.dtype, device=row_ind.device
                        ),
                    ]
                )
                col_ind_full = torch.cat(
                    [
                        col_ind,
                        torch.tensor(
                            [j], dtype=col_ind.dtype, device=col_ind.device
                        ),
                    ]
                )
                if self.update_with_lap_solutions:
                    best_cost, best_permutation = update_best_permutation(
                        best_cost,
                        best_permutation,
                        row_ind_full,
                        col_ind_full,
                        cost1,
                        cost2,
                        linear,
                        constant,
                    )

                # update the u, v values and the leader matrix
                us[i, j] += u
                vs[i, j] += v

                if time() - start_time > timelimit:
                    break

                leaders[i, j] = u.sum() + v.sum()

            if improvement < min(
                eps, eps * superleader.abs(), eps * best_cost.abs()
            ):
                steps_without_improvement += 1
            else:
                steps_without_improvement = 0

            if (
                steps_without_improvement > self.max_steps_without_improvement
                or time() - start_time > timelimit
            ):
                break

            eps_auction *= 0.9

        if self.needs_epsilon:
            u, v, row_ind, col_ind = self.matching_algorithm(
                leaders, eps=eps_auction
            )
        else:
            u, v, row_ind, col_ind = self.matching_algorithm(leaders)
        best_cost, best_permutation = update_best_permutation(
            best_cost,
            best_permutation,
            row_ind,
            col_ind,
            cost1,
            cost2,
            linear,
            constant,
        )

        return best_permutation, optimal, best_cost.item(), superleader.item()

    def __repr__(self):
        """
        Returns a string representation of the solver instance.
        """
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items() if k not in ['matching_algorithm', 'needs_epsilon_dict']])})"
