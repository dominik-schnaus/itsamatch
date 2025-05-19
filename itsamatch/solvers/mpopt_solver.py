from warnings import warn

import torch
from torch import Tensor

try:
    from mpopt import qap
except ImportError:
    print("mpopt import skipped.")

from .qap_solver import QAPSolver


class MPOpt(QAPSolver):
    """
    MPOpt solver for the Quadratic Assignment Problem (QAP).

    This solver uses the MPOpt library, which implements a message passing
    algorithm for solving QAPs. It is suitable for factorized cost matrices.
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
        n = cost1.shape[0]

        if timelimit is not None:
            warn("timelimit is not supported in the MPOpt solver.")

        # Initialize linear and constant terms if not provided
        if linear is None:
            linear = torch.zeros_like(cost1)

        if constant is None:
            constant = torch.zeros(1, device=cost1.device)

        # Construct the quadratic cost matrix from factorized inputs
        quadratic_cost = cost1[:, None, :, None] * cost2[None, :, None, :]

        # Shift costs to be non-positive for MPOpt, adjusting the constant term
        # This is a common technique to ensure MPOpt's internal assumptions are met
        # while preserving the original problem's solution.
        constant += (quadratic_cost.max() + 1) * n**2
        constant += (linear.max() + 1) * n

        quadratic_cost -= quadratic_cost.max() + 1
        linear -= linear.max() + 1

        # Initialize the MPOpt model
        model = qap.Model(
            no_left=n, no_right=n, no_assignments=n**2, no_edges=n**4 - n**2
        )
        # Add assignment costs (unary terms)
        for i in range(n):
            for j in range(n):
                model.add_assignment(
                    i * n + j,  # Unique ID for assignment (i, j)
                    i,  # Left node
                    j,  # Right node
                    quadratic_cost[i, j, i, j].item()
                    + linear[i, j].item(),  # Cost
                )
        # Add pairwise costs (binary terms)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):  # noqa: E741
                        id_assignment1 = i * n + j
                        id_assignment2 = k * n + l
                        # Ensure each edge is added once and no self-loops or conflicting assignments
                        if (
                            id_assignment1 < id_assignment2
                            and i
                            != k  # Assignments must involve different left nodes
                            and j
                            != l  # Assignments must involve different right nodes
                        ):
                            model.add_edge(
                                id_assignment1,
                                id_assignment2,
                                quadratic_cost[i, j, k, l].item()
                                + quadratic_cost[
                                    k, l, i, j
                                ].item(),  # Symmetric cost
                            )

        # Decompose the model for the solver
        deco = qap.ModelDecomposition(
            model, with_uniqueness=True, unary_side="left"
        )
        # Construct and run the solver
        solver = qap.construct_solver(deco)
        solver.run(
            qap.DEFAULT_BATCH_SIZE,
            qap.DEFAULT_MAX_BATCHES,
            qap.DEFAULT_GREEDY_GENERATIONS,
        )

        # Extract primal solution (the assignment)
        primals = qap.extract_primals(deco, solver)

        # Calculate lower bound and objective value, adjusting for the cost shift
        lower_bound = solver.lower_bound() + constant
        objective_value = primals.evaluate() + constant
        # Check for optimality
        optimal = objective_value <= lower_bound + 1e-6

        # Convert labeling to a tensor
        labeling = torch.as_tensor(
            primals.labeling, dtype=torch.long, device=cost1.device
        )

        return (
            labeling,
            optimal.item(),
            objective_value.item(),
            lower_bound.item(),
        )
