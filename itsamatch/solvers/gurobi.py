"""This module provides a Gurobi-based solver for the Quadratic Assignment Problem (QAP)."""

import gurobipy as gp
import torch
from gurobipy import GRB
from torch import Tensor

from .qap_solver import QAPSolver


class Gurobi(QAPSolver):
    """
    Gurobi solver for the Quadratic Assignment Problem (QAP).
    """

    factorized: bool = False

    def __init__(self, relaxed: bool = False):
        """
        Initialize the Gurobi solver.

        Args:
            relaxed (bool): If True, the solver will use continuous variables instead of binary variables.
        """
        super().__init__()
        self.relaxed = relaxed

    def solve(
        self,
        cost: Tensor,
        timelimit: float | None = None,
    ) -> tuple[Tensor, bool, float, float | None]:
        """
        Solve the QAP for the given inputs.

        Args:
            cost (Tensor): Cost matrix.
            timelimit (float, optional): Time limit for the solver. Defaults to None.

        Returns:
            permutation (Tensor): The permutation that minimizes the cost.
            optimal (bool): Whether the solution is optimal.
            cost (float): The minimum cost.
            bound (float, optional): The lower bound of the cost.
        """
        # Check if the cost matrix is valid
        self._check_size(cost)
        size = cost.shape[0]

        # Create a new Gurobi model
        m = gp.Model("quadratic assignment")
        # Suppress Gurobi output
        m.Params.OutputFlag = 0

        # Set NonConvex parameter to 2 to handle non-convex quadratic objectives
        m.Params.NonConvex = 2

        # Create decision variables: T[i, j] = 1 if item i is assigned to position j, 0 otherwise
        T = m.addMVar(
            shape=(size, size),
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS
            if self.relaxed
            else GRB.BINARY,  # Use continuous or binary variables based on the 'relaxed' flag
            name="T",
        )

        # Set the objective function: minimize sum(cost[i, j, k, l] * T[i, j] * T[k, l])
        m.setObjective(
            gp.quicksum(
                cost[i, j, k, l] * T[i, j] * T[k, l]
                for i in range(size)
                for j in range(size)
                for k in range(size)
                for l in range(size)  # noqa: E741
            ),
            GRB.MINIMIZE,
        )

        # Add row constraints: each item must be assigned to exactly one position
        m.addConstrs((T[i, :].sum() == 1 for i in range(size)), name="t1")

        # Add column constraints: each position must be occupied by exactly one item
        m.addConstrs((T[:, j].sum() == 1 for j in range(size)), name="t2")

        # Set time limit for the solver if provided
        if timelimit is not None:
            m.Params.TimeLimit = timelimit

        # Optimize the model
        m.optimize()

        # Retrieve the objective value (cost)
        objective_value = m.objVal
        # Retrieve the lower bound of the cost
        bound = m.ObjBound
        # Check if the solution is optimal
        is_optimal = m.status == GRB.OPTIMAL
        # Convert the Gurobi solution to a PyTorch tensor representing the permutation
        prediction = torch.as_tensor(T.X).to(cost).argmax(dim=1)

        # Dispose of the Gurobi model to free resources
        m.dispose()

        return prediction, is_optimal, objective_value, bound
