"""Abstract base class for Quadratic Assignment Problem (QAP) solvers."""

from abc import ABC, abstractmethod

from torch import Tensor


class QAPSolver(ABC):
    """
    Abstract class for Quadratic Assignment Problem (QAP) solvers.
    """

    factorized: bool

    @abstractmethod
    def solve(
        self,
        *args,
        **kwargs,
    ) -> tuple[Tensor, bool, float, float | None]:
        """
        Solve the QAP for the given inputs.

        Args (factorized):
            cost1 (Tensor): First factorized cost matrix.
            cost2 (Tensor): Second factorized cost matrix.
            linear (Tensor, optional): Linear term. Defaults to None.
            constant (Tensor, optional): Constant term. Defaults to None.
            timelimit (float, optional): Time limit for the solver. Defaults to None.

        Args (non-factorized):
            cost (Tensor): Cost matrix.
            timelimit (float, optional): Time limit for the solver. Defaults to None.

        Returns:
            permutation (Tensor): The permutation that minimizes the cost.
            optimal (bool): Whether the solution is optimal.
            cost (float): The minimum cost.
            bound (float, optional): The lower bound of the cost.
        """
        raise NotImplementedError

    def _check_size_factorized(
        self,
        cost1: Tensor,
        cost2: Tensor,
        linear: Tensor = None,
        constant: Tensor = None,
    ):
        """
        Check the size of the factorized cost matrices and optional linear
        and constant terms.

        Args:
            cost1 (Tensor): First factorized cost matrix.
            cost2 (Tensor): Second factorized cost matrix.
            linear (Tensor, optional): Linear term. Defaults to None.
            constant (Tensor, optional): Constant term. Defaults to None.
        """
        assert cost1.ndim == 2, "cost1 must be a 2D matrix"
        assert cost2.ndim == 2, "cost2 must be a 2D matrix"
        assert linear is None or linear.ndim == 2, "linear must be a 2D matrix"
        assert constant is None or constant.numel() == 1, (
            "constant must be a scalar"
        )

        assert cost1.shape[0] == cost1.shape[1], "cost1 must be a square matrix"
        assert cost2.shape[0] == cost2.shape[1], "cost2 must be a square matrix"
        assert cost1.shape[0] == cost2.shape[0], (
            "cost1 and cost2 must have the same size"
        )
        if linear is not None:
            assert cost1.shape[0] == linear.shape[0], (
                "linear must have the same size as cost1"
            )

    def _check_size(
        self,
        cost: Tensor,
    ):
        """
        Check the size of the cost matrix.

        Args:
            cost (Tensor): Cost tensor.
        """
        assert cost.ndim == 4, "cost must be a 4D tensor"

        assert (
            cost.shape[0] == cost.shape[1] == cost.shape[2] == cost.shape[3]
        ), "each dimension of cost must be the same"

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    def __str__(self):
        return self.__class__.__name__
