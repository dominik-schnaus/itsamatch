"""
Solves the Quadratic Assignment Problem (QAP) using Optimal Transport (OT) methods,
specifically Gromov-Wasserstein distance.
"""

from warnings import warn

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

try:
    from ot import gromov
except ImportError:
    warn(
        "Optimal transport requires the 'POT' package. "
        "Please install it with 'pip install POT'."
    )

from ..utils import get_seeds
from .qap_solver import QAPSolver


def optimal_transport(
    cost1: np.ndarray,
    cost2: np.ndarray,
    method: str,
    random_G0: bool | None = True,
    rng: int | None = None,
    **method_kwargs,
) -> tuple[torch.Tensor, float]:
    """
    Computes the Gromov-Wasserstein optimal transport plan and extracts a permutation.

    Args:
        cost1 (np.ndarray): The first cost matrix (e.g., intra-domain distances/similarities).
        cost2 (np.ndarray): The second cost matrix.
        method (str): The Gromov-Wasserstein solver method from `ot.gromov`
                      (e.g., 'entropic_gromov_wasserstein').
        random_G0 (bool, optional): Whether to initialize the transport plan G0 randomly.
                                    If False, G0 is initialized to None (POT's default).
                                    Defaults to True.
        rng (int, optional): Random seed for G0 initialization if `random_G0` is True.
                             Defaults to None.
        **method_kwargs: Additional keyword arguments for the chosen OT method.

    Returns:
        prediction (torch.Tensor): The predicted permutation as a tensor of indices.
        gw_dist (float): The Gromov-Wasserstein distance.
    """
    source_size = cost1.shape[0]
    target_size = cost2.shape[0]

    # Initialize the transport plan G0
    if random_G0:
        if rng is None:
            G0 = torch.rand(source_size, target_size).double()
        else:
            generator = torch.Generator()
            generator.manual_seed(rng)
            G0 = torch.rand(
                source_size, target_size, generator=generator
            ).double()
        # Normalize G0 to be doubly stochastic (approximately)
        # This step is crucial for some OT solvers or to guide the solver.
        while not (
            torch.allclose(
                G0.sum(dim=1, keepdim=True),
                torch.ones(G0.shape[0]).to(G0) / source_size,
                rtol=1e-8,
                atol=1e-9,
            )
            and torch.allclose(
                G0.sum(dim=0, keepdim=True),
                torch.ones(G0.shape[1]).to(G0) / target_size,
                rtol=1e-8,
                atol=1e-9,
            )
        ):
            G0 = G0 / (source_size * G0.sum(dim=1, keepdim=True))
            G0 = G0 / (target_size * G0.sum(dim=0, keepdim=True))
    else:
        G0 = None  # Use POT's default initialization for G0

    # Get the specified Gromov-Wasserstein solver function from POT library
    method_function = getattr(gromov, method)

    # Compute the optimal transport matrix using the chosen method
    transport_matrix, log = method_function(
        cost1,
        cost2,
        loss_fun="square_loss",  # Use squared Euclidean distance for differences
        symmetric=True,  # Assume cost matrices are symmetric
        verbose=False,
        log=True,  # Return log for GW distance
        G0=G0.numpy() if G0 is not None else None,  # Initial transport plan
        amijo=True,  # Use Armijo line search for optimization
        max_iter=1e3,  # Maximum number of iterations
        tol_rel=1e-10,  # Relative tolerance for convergence
        tol_abs=1e-10,  # Absolute tolerance for convergence
        **method_kwargs,
    )

    # Extract the permutation from the transport matrix using linear sum assignment
    # We maximize because the transport matrix entries represent matching scores.
    _, prediction = linear_sum_assignment(transport_matrix, maximize=True)
    prediction = torch.as_tensor(prediction, dtype=torch.long)

    return prediction, log["gw_dist"]


class OptimalTransport(QAPSolver):
    """
    Optimal Transport (OT) based solver for the Quadratic Assignment Problem (QAP).
    """

    factorized: bool = True

    def __init__(
        self,
        method: str = "entropic_gromov_wasserstein",
        method_kwargs: dict = None,
        num_repetitions: int = 1,
    ):
        """
        Initialize the OptimalTransport solver.

        Args:
            method (str, optional): The Gromov-Wasserstein solver method from `ot.gromov`.
                                    Defaults to 'entropic_gromov_wasserstein'.
            method_kwargs (dict, optional): Additional keyword arguments for the OT method.
                                            Defaults to None.
            num_repetitions (int, optional): Number of times to run the OT solver,
                                             each time (except the first) with a random
                                             initialization for G0. The best result is kept.
                                             Defaults to 1.
        """
        super().__init__()
        self.method = method
        self.method_kwargs = method_kwargs if method_kwargs is not None else {}
        self.num_repetitions = num_repetitions

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

        if timelimit is not None:
            warn("timelimit is not supported in the optimal transport solver.")

        if linear is not None:
            warn(
                "linear is not supported in the optimal transport solver. "
                "The solver will ignore the linear term."
            )

        # Convert tensors to NumPy arrays for POT library
        cost1_np = cost1.cpu().numpy()
        cost2_np = cost2.cpu().numpy()

        # The OT solver with 'square_loss' minimizes sum (cost1_ik - cost2_jl)^2 P_ij P_kl.
        # This is equivalent to minimizing sum (- cost1_ik) cost2_jl P_ij P_kl.
        cost1_np = -cost1_np

        min_gw_dist = torch.inf
        best_prediction = None

        seeds = get_seeds(num_seeds=self.num_repetitions)
        # Run the OT solver multiple times if requested, using random G0 for repetitions
        for i in range(self.num_repetitions):
            # Use POT's default G0 for the first run, random G0 for subsequent runs
            use_random_G0 = i > 0

            current_prediction, current_gw_dist = optimal_transport(
                cost1_np,
                cost2_np,
                self.method,
                random_G0=use_random_G0,
                rng=seeds[i],
                **self.method_kwargs,
            )

            # Keep the prediction that results in the minimum Gromov-Wasserstein distance
            if current_gw_dist < min_gw_dist:
                min_gw_dist = current_gw_dist
                best_prediction = current_prediction

        # Calculate the QAP objective value sum_{i,j} cost1[i,j] * cost2[P[i], P[j]]
        qap_cost = (
            (cost1 * cost2[best_prediction][:, best_prediction]).sum().item()
        )

        if constant is not None:
            qap_cost += constant.item()

        # OT solvers for QAP are generally heuristic, so optimality is not guaranteed.
        return best_prediction, False, qap_cost, None
