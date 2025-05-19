"""
Auction algorithm for the Linear Assignment Problem (LAP).

This module implements the auction algorithm as described by Bertsekas.
The implementation is based on the version from https://github.com/bkj/auction-lap/
and the original paper:
"Auction Algorithms for Network Flow Problems: A Tutorial Introduction".
(Available at: https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf;sequence=1)
"""

from __future__ import division, print_function

import torch

from itsamatch.utils import seed_everything


def auction_lap(X, eps=None, minimize=False):
    """
    Solves the Linear Assignment Problem (LAP) using the auction algorithm.

    The algorithm simulates an auction where 'persons' (rows) bid for 'objects'
    (columns). The goal is to find an assignment that either maximizes the total
    profit or minimizes the total cost.

    Args:
        X (torch.Tensor): A 2D tensor representing the cost/profit matrix.
            `X[i, j]` is the value of assigning person `i` to object `j`.
            The matrix should ideally be square (n-by-n). If not, it implies
            some persons or objects might not be assigned. This implementation
            assumes a square matrix or at least `X.shape[0] <= X.shape[1]`.
        eps (float, optional): The bidding increment, also known as epsilon.
            This value influences the precision and convergence speed.
            A smaller epsilon generally leads to a solution closer to the true
            optimum but may require more iterations. If None, it defaults to
            `1 / X.shape[0]`.
        minimize (bool, optional): If True, the problem is treated as a cost
            minimization problem. If False (default), it's a profit maximization
            problem.

    Returns:
        tuple: A tuple `(u, v, row_ind, col_ind)` where:
            - u (torch.Tensor): The final values (dual variables) for persons (rows).
            - v (torch.Tensor): The final prices (dual variables) for objects (columns).
            - row_ind (torch.Tensor): Tensor of row indices for the assignment.
                                      Corresponds to `torch.arange(X.shape[0])`.
            - col_ind (torch.Tensor): Tensor of column indices indicating the object
                                      assigned to each person. `col_ind[i]` is the
                                      object assigned to person `i`.
    """
    # Set epsilon if not provided
    eps = 1 / X.shape[0] if eps is None else eps
    # If minimizing, negate the profit matrix to turn it into a maximization problem
    if minimize:
        X = -X

    # Initialization
    # cost: current prices of objects (columns)
    cost = torch.zeros(1, X.shape[1], device=X.device, dtype=X.dtype)
    # curr_ass: current assignment of persons (rows) to objects (columns).
    # -1 indicates unassigned.
    curr_ass = torch.zeros(X.shape[0], device=X.device, dtype=torch.long) - 1
    # bids: temporary storage for bids made by persons in an iteration
    bids = torch.zeros(*X.shape, device=X.device, dtype=X.dtype)

    counter = 0  # Iteration counter
    # Main auction loop: continues as long as there are unassigned persons
    while (curr_ass == -1).any():
        counter += 1

        # Bidding phase
        # Identify unassigned persons
        unassigned_persons_indices = (curr_ass == -1).nonzero().squeeze()

        # For each unassigned person, calculate the "value" of each object:
        # value = profit_from_object - price_of_object
        # X[unassigned] gives rows of X corresponding to unassigned persons
        value_for_unassigned = X[unassigned_persons_indices] - cost

        # Find the best and second-best objects for each unassigned person
        # topk(2) returns (values, indices)
        top_values, top_indices = value_for_unassigned.topk(2, dim=1)

        # Index of the most desirable object for each unassigned person
        best_object_indices = top_indices[:, 0]
        # Value of the best object
        best_object_values = top_values[:, 0]
        # Value of the second-best object (or a very small number if only one object)
        second_best_object_values = top_values[:, 1]

        # Calculate bid increments:
        # bid = value_of_best_object - value_of_second_best_object + epsilon
        # This is the amount the person is willing to raise the price of their best object
        bid_increments = best_object_values - second_best_object_values + eps

        # Prepare bids tensor for current unassigned persons
        current_bids = bids[unassigned_persons_indices]
        current_bids.zero_()  # Clear previous bids for these persons

        # Handle single unassigned person case (unassigned_persons_indices is a scalar)
        if unassigned_persons_indices.dim() == 0:
            # The single unassigned person bids for their best object
            # Update the price of the chosen object
            cost[:, best_object_indices] += bid_increments
            # Unassign any person currently assigned to this object
            curr_ass[(curr_ass == best_object_indices).nonzero()] = -1
            # Assign the object to the current bidder
            curr_ass[unassigned_persons_indices] = best_object_indices
        else:  # Multiple unassigned persons
            # Place bids: each unassigned person bids for their best object
            current_bids.scatter_(
                dim=1,  # Scatter along columns (objects)
                index=best_object_indices.contiguous().view(
                    -1, 1
                ),  # Indices of objects to bid on
                src=bid_increments.view(-1, 1),  # Bid amounts
            )

            # Assignment phase
            # Identify objects that have received at least one bid
            # (bids_ > 0).int().sum(dim=0) sums bids per object
            objects_with_bids_indices = (
                (current_bids > 0).int().sum(dim=0).nonzero()
            )
            if (
                objects_with_bids_indices.numel() == 0
            ):  # No bids were placed, something is wrong or all assigned
                continue

            # For each object that received bids, find the highest bid and the bidder
            highest_bids, highest_bidder_relative_indices = current_bids[
                :, objects_with_bids_indices
            ].max(dim=0)

            # Convert relative bidder indices (among unassigned) to absolute person indices
            highest_bidders_absolute_indices = unassigned_persons_indices[
                highest_bidder_relative_indices.squeeze()
            ]

            # Increase the price of objects by the highest bid amount
            cost[:, objects_with_bids_indices] += highest_bids

            # Unassign persons who were previously assigned to these objects
            # (curr_ass.view(-1, 1) == objects_with_bids_indices.view(1, -1)) creates a boolean matrix
            # .sum(dim=1) checks if a person is assigned to any of the objects_with_bids
            # .nonzero() gets indices of such persons
            persons_to_unassign_indices = (
                (curr_ass.view(-1, 1) == objects_with_bids_indices.view(1, -1))
                .sum(dim=1)
                .nonzero()
            )
            curr_ass[persons_to_unassign_indices] = -1

            # Assign objects to the new highest bidders
            curr_ass[highest_bidders_absolute_indices] = (
                objects_with_bids_indices.squeeze()
            )

    # Calculate dual variables u (for rows/persons) and v (for columns/objects)
    # u_i = max_j (profit_ij - price_j)
    # v_j = price_j
    u = -(-X + cost).min(dim=1).values  # Equivalent to max_j (X_ij - cost_j)
    v = cost[0]  # Prices of objects

    # If minimizing, convert dual variables back
    if minimize:
        u, v = -u, -v

    # Return dual variables and the assignment (row_ind, col_ind)
    return (
        u,
        v,
        torch.arange(X.shape[0], device=X.device, dtype=torch.long).to(
            curr_ass.device
        ),
        curr_ass,
    )


if __name__ == "__main__":
    seed_everything(42)
    num_workers = num_jobs = 5
    cost_matrix = torch.randint(
        0, 10, (num_workers, num_jobs), dtype=torch.float
    )
    eps = 1e-4
    u, v, row_ind, col_ind = auction_lap(cost_matrix, eps=eps, minimize=False)
    print(u, v, row_ind, col_ind)
    print(cost_matrix[row_ind, col_ind].sum())
    print(u.sum() + v.sum())
