import sys
from os.path import dirname, join
from typing import Optional

import torch
from torch import Tensor
from torch.utils.cpp_extension import load

module = load(
    name="forward_reverse_auction",
    sources=[join(dirname(__file__), "forward_reverse_auction.cpp")],
)
sys.modules["forward_reverse_auction"] = module


def get_forward_backward_auction_cpp(minimize: bool = False):
    def forward_reverse_auction_cpp(
        cost_matrix: Tensor, eps: Optional[float] = None
    ):
        if eps is None:
            eps = (cost_matrix.max() - cost_matrix.min()) / 50
        elif not isinstance(eps, Tensor):
            eps = torch.tensor(
                eps, dtype=cost_matrix.dtype, device=cost_matrix.device
            )
        u, v, row_ind, col_ind = module.forward_reverse_auction(
            cost_matrix,
            eps,
            minimize,
        )
        return u, v, row_ind, col_ind

    return forward_reverse_auction_cpp


def _forward_auction(
    cost_matrix: Tensor,
    price,
    col2row,
    row2col,
    bids,
    epsilon,
):
    unassigned = torch.where(row2col == -1)[0]
    value = cost_matrix[unassigned] - price.unsqueeze(0)
    top_value, top_idx = value.topk(2, dim=1)
    first_idx = top_idx[:, 0]
    first_value, second_value = top_value[:, 0], top_value[:, 1]
    bid_increments = first_value - second_value + epsilon
    bids_ = bids[unassigned]
    bids_.zero_()
    bids_.scatter_(
        dim=1,
        index=first_idx.unsqueeze(1),
        src=bid_increments.unsqueeze(1),
    )
    high_bids, high_bidders = bids_.max(dim=0)
    have_bids = torch.where(high_bids > 0)[0]
    row_indices = unassigned[high_bidders[have_bids]]
    price += high_bids
    col2row[have_bids] = row_indices
    row2col[torch.isin(row2col, have_bids)] = -1
    row2col[row_indices] = have_bids
    return price, col2row, row2col


def _reverse_auction(
    cost_matrix: Tensor,
    profit,
    col2row,
    row2col,
    bids,
    epsilon,
):
    unassigned = torch.where(col2row == -1)[0]
    value = cost_matrix[:, unassigned] - profit.unsqueeze(1)
    top_value, top_idx = value.topk(2, dim=0)
    first_idx = top_idx[0]
    first_value, second_value = top_value[0], top_value[1]
    bid_increments = first_value - second_value + epsilon
    bids_ = bids[:, unassigned]
    bids_.zero_()
    bids_.scatter_(
        dim=0,
        index=first_idx.unsqueeze(0),
        src=bid_increments.unsqueeze(0),
    )
    high_bids, high_bidders = bids_.max(dim=1)
    have_bids = torch.where(high_bids > 0)[0]
    row_indices = unassigned[high_bidders[have_bids]]
    profit += high_bids
    row2col[have_bids] = row_indices
    col2row[torch.isin(col2row, have_bids)] = -1
    col2row[row_indices] = have_bids
    return profit, col2row, row2col


def fra(cost_matrix, epsilon, minimize=False):
    if minimize:
        cost_matrix = -cost_matrix

    n = cost_matrix.shape[0]

    profit = torch.zeros(n)
    price = torch.zeros(n)
    row2col = -torch.ones(n, dtype=torch.long)
    col2row = -torch.ones(n, dtype=torch.long)
    bids = torch.zeros(n, n)
    iteration = 0
    while (row2col == -1).any():
        iteration += 1
        num_assigned = (row2col != -1).sum()
        while (row2col != -1).sum() <= num_assigned:
            price, col2row, row2col = _forward_auction(
                cost_matrix, price, col2row, row2col, bids, epsilon
            )
        profit = (cost_matrix - price.unsqueeze(0)).max(dim=1).values
        if (row2col != -1).all():
            break
        num_assigned = (row2col != -1).sum()
        while (row2col != -1).sum() <= num_assigned:
            profit, col2row, row2col = _reverse_auction(
                cost_matrix, profit, col2row, row2col, bids, epsilon
            )
        price = (cost_matrix - profit.unsqueeze(1)).max(dim=0).values
        if (row2col != -1).all():
            break
    if minimize:
        profit = -profit
        price = -price
    return (
        profit,
        price,
        torch.arange(n, dtype=row2col.dtype, device=row2col.device),
        row2col,
    )


def forward_reverse_auction(cost_matrix, epsilon, minimize=False):
    if minimize:
        cost_matrix = -cost_matrix

    n = cost_matrix.size(0)

    profit = torch.zeros(n)
    price = torch.zeros(n)
    row2col = -torch.ones(n, dtype=torch.long)
    col2row = -torch.ones(n, dtype=torch.long)
    bids = torch.zeros(n, n)
    iteration = 0

    high_bids = torch.empty(n)
    high_bidders = torch.empty(n, dtype=torch.long)

    while (row2col == -1).any():
        iteration += 1
        num_assigned = (row2col != -1).sum().item()

        while (row2col != -1).sum().item() <= num_assigned:
            # Forward Auction
            unassigned = torch.where(row2col == -1)[0]
            value = cost_matrix[unassigned] - price.unsqueeze(0)
            top_value, top_idx = value.topk(2, 1)
            first_idx = top_idx[:, 0]
            first_value = top_value[:, 0]
            second_value = top_value[:, 1]
            bid_increments = first_value - second_value + epsilon
            bids.zero_()
            bids[unassigned.unsqueeze(1), first_idx.unsqueeze(1)] = (
                bid_increments.unsqueeze(1)
            )
            torch.max(bids, 0, out=(high_bids, high_bidders))
            have_bids = torch.where(high_bids > 0)[0]
            row_indices = high_bidders[have_bids]
            price += high_bids
            col2row[have_bids] = row_indices
            row2col[torch.isin(row2col, have_bids)] = -1
            row2col[row_indices] = have_bids

        # Profit Calculation
        profit = (cost_matrix - price.unsqueeze(0)).max(1)[0]

        if (row2col != -1).all():
            break

        num_assigned = (row2col != -1).sum().item()

        while (row2col != -1).sum().item() <= num_assigned:
            # Reverse Auction
            unassigned = torch.where(col2row == -1)[0]
            value = cost_matrix[:, unassigned] - profit.unsqueeze(1)
            top_value, top_idx = value.topk(2, 0)
            first_idx = top_idx[0]
            first_value = top_value[0]
            second_value = top_value[1]
            bid_increments = first_value - second_value + epsilon
            bids.zero_()
            bids[first_idx.unsqueeze(0), unassigned.unsqueeze(0)] = (
                bid_increments.unsqueeze(0)
            )
            torch.max(bids, 1, out=(high_bids, high_bidders))
            have_bids = torch.where(high_bids > 0)[0]
            row_indices = high_bidders[have_bids]
            profit += high_bids
            row2col[have_bids] = row_indices
            col2row[torch.isin(col2row, have_bids)] = -1
            col2row[row_indices] = have_bids

        # Price Calculation
        price = (cost_matrix - profit.unsqueeze(1)).max(0)[0]

        if (row2col != -1).all():
            break

    if minimize:
        profit.neg_()
        price.neg_()

    row_indices = torch.arange(n)
    return profit, price, row_indices, row2col


if __name__ == "__main__":
    cost_matrix = torch.tensor(
        [
            [42.0, 67.0, 76.0, 14.0, 26.0, 35.0, 20.0, 24.0, 50.0, 13.0],
            [78.0, 14.0, 10.0, 54.0, 31.0, 72.0, 15.0, 95.0, 67.0, 6.0],
            [49.0, 76.0, 73.0, 11.0, 99.0, 13.0, 41.0, 69.0, 87.0, 19.0],
            [72.0, 80.0, 75.0, 29.0, 33.0, 64.0, 39.0, 76.0, 32.0, 10.0],
            [86.0, 22.0, 77.0, 19.0, 7.0, 23.0, 43.0, 94.0, 93.0, 77.0],
            [70.0, 9.0, 70.0, 39.0, 86.0, 99.0, 15.0, 84.0, 78.0, 8.0],
            [66.0, 30.0, 40.0, 60.0, 70.0, 61.0, 23.0, 20.0, 11.0, 61.0],
            [77.0, 89.0, 84.0, 53.0, 48.0, 9.0, 83.0, 7.0, 58.0, 91.0],
            [14.0, 91.0, 36.0, 3.0, 82.0, 90.0, 89.0, 28.0, 55.0, 33.0],
            [27.0, 47.0, 65.0, 89.0, 41.0, 45.0, 61.0, 39.0, 61.0, 64.0],
        ]
    )

    eps = 0.1

    u, v, row_ind, col_ind = fra(cost_matrix, eps, minimize=True)
    print(u, v, row_ind, col_ind)
    print(u.sum() + v.sum())
    print(cost_matrix[row_ind, col_ind].sum())

    profit, price, row2col, row_indices = forward_reverse_auction(
        cost_matrix, epsilon=eps, minimize=True
    )
    print(profit, price, row2col, row_indices)
    print(profit.sum() + price.sum())
    print(cost_matrix[row2col, row_indices].sum())

    auction_fairseq = get_forward_backward_auction_cpp(minimize=True)
    u, v, row_ind, col_ind = auction_fairseq(cost_matrix, eps)
    print(u, v, row_ind, col_ind)
    print(u.sum() + v.sum())
    print(cost_matrix[row_ind, col_ind].sum())
