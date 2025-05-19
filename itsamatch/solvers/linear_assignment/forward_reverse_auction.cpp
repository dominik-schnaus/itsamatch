#include <torch/extension.h>
#include <iostream>
#include <tuple>

using namespace torch::indexing;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_reverse_auction(torch::Tensor cost_matrix, torch::Tensor epsilon,
                        bool minimize = false) {
  if (minimize) {
    cost_matrix = -cost_matrix;
  }

  int n = cost_matrix.size(0);

  torch::Tensor profit = torch::zeros({n}, cost_matrix.options());
  torch::Tensor price = torch::zeros({n}, cost_matrix.options());
  torch::Tensor row2col = -torch::ones({n}, torch::kLong).to(cost_matrix.device());
  torch::Tensor col2row = -torch::ones({n}, torch::kLong).to(cost_matrix.device());
  torch::Tensor bids = torch::zeros({n, n}, cost_matrix.options());
  int iteration = 0;

  // Pre-allocate memory for high_bids and high_bidders
  torch::Tensor high_bids = torch::empty({n}, cost_matrix.options());
  torch::Tensor high_bidders = torch::empty({n}, torch::kLong).to(cost_matrix.device());

  while ((row2col == -1).any().item<bool>()) {
    iteration += 1;
    int num_assigned = (row2col != -1).sum().item<int>();

    while ((row2col != -1).sum().item<int>() <= num_assigned) {
      // Forward Auction
      torch::Tensor unassigned = torch::where(row2col == -1)[0];
      torch::Tensor value =
          cost_matrix.index({unassigned}) - price.unsqueeze(0);
      torch::Tensor top_value, top_idx;
      std::tie(top_value, top_idx) = value.topk(2, 1);
      torch::Tensor first_idx = top_idx.index({Slice(None), 0});
      torch::Tensor first_value = top_value.index({Slice(None), 0});
      torch::Tensor second_value = top_value.index({Slice(None), 1});
      torch::Tensor bid_increments = first_value - second_value + epsilon;
      bids.zero_();
      bids.index_put_({unassigned.unsqueeze(1), first_idx.unsqueeze(1)},
                       bid_increments.unsqueeze(1));
      // Use the out parameter of max to store the results directly into high_bids and high_bidders
      torch::max_out(high_bids, high_bidders, bids, 0);
      torch::Tensor have_bids = torch::where(high_bids > 0)[0];
      torch::Tensor row_indices = high_bidders.index({have_bids});
      price += high_bids;
      col2row.index_put_({have_bids}, row_indices);
      row2col.index_put_({torch::isin(row2col, have_bids)}, -1);
      row2col.index_put_({row_indices}, have_bids);
    }

    // Profit Calculation
    profit = std::get<0>((cost_matrix - price.unsqueeze(0)).max(1));

    if ((row2col != -1).all().item<bool>()) {
      break;
    }

    num_assigned = (row2col != -1).sum().item<int>();

    while ((row2col != -1).sum().item<int>() <= num_assigned) {
      // Reverse Auction
      torch::Tensor unassigned = torch::where(col2row == -1)[0];
      torch::Tensor value =
          cost_matrix.index({Slice(), unassigned}) - profit.unsqueeze(1);
      torch::Tensor top_value, top_idx;
      std::tie(top_value, top_idx) = value.topk(2, 0);
      torch::Tensor first_idx = top_idx.index({0});
      torch::Tensor first_value = top_value.index({0});
      torch::Tensor second_value = top_value.index({1});
      torch::Tensor bid_increments = first_value - second_value + epsilon;
      bids.zero_();
      bids.index_put_({first_idx.unsqueeze(0), unassigned.unsqueeze(0)},
                       bid_increments.unsqueeze(0));
      // Use the out parameter of max to store the results directly into high_bids and high_bidders
      torch::max_out(high_bids, high_bidders, bids, 1);
      torch::Tensor have_bids = torch::where(high_bids > 0)[0];
      torch::Tensor row_indices = high_bidders.index({have_bids});
      profit += high_bids;
      row2col.index_put_({have_bids}, row_indices);
      col2row.index_put_({torch::isin(col2row, have_bids)}, -1);
      col2row.index_put_({row_indices}, have_bids);
    }

    // Price Calculation
    price = std::get<0>((cost_matrix - profit.unsqueeze(1)).max(0));

    if ((row2col != -1).all().item<bool>()) {
      break;
    }
  }

  if (minimize) {
    profit.neg_();
    price.neg_();
  }

  torch::Tensor row_indices = torch::arange(n, row2col.options());
  return std::make_tuple(profit, price, row_indices, row2col);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_reverse_auction", &forward_reverse_auction, "Forward reverse auction");
}