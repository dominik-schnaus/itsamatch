import torch
from scipy.optimize import linear_sum_assignment


def hungarian_matching(C, **kwargs):
    """
    Solves the linear assignment problem using the Hungarian algorithm.

    This function finds the optimal assignment of rows to columns in a cost matrix C,
    such that the sum of costs of the assigned elements is minimized.
    It returns the dual variables u and v, and the row and column assignments.

    Args:
        C (torch.Tensor): The cost matrix of shape (m, n).
        **kwargs: Additional keyword arguments (not used in this implementation).

    Returns:
        tuple: A tuple containing:
            - u (torch.Tensor): The dual variables for rows.
            - v (torch.Tensor): The dual variables for columns.
            - row_ind (torch.Tensor): The row indices of the optimal assignment.
            - col_ind (torch.Tensor): The column indices of the optimal assignment.
            Returns None if no solution is found (e.g., due to all-infinite costs).
    """
    inf = torch.inf if C.dtype.is_floating_point else torch.iinfo(C.dtype).max

    m = C.shape[0]
    n = C.shape[1]
    transpose = False

    # Ensure that the number of rows is less than or equal to the number of columns
    if m > n:
        C = C.T
        m, n = n, m
        transpose = True

    # Initialize dual variables u and v
    u = torch.zeros(m, dtype=C.dtype, device=C.device)
    v = torch.zeros(n, dtype=C.dtype, device=C.device)

    # Initialize assignment arrays
    col4row = -torch.ones(
        m, dtype=torch.long
    )  # Stores the column assigned to each row
    row4col = -torch.ones(
        n, dtype=torch.long
    )  # Stores the row assigned to each column

    # Set of available columns
    AC = set(list(range(n)))

    # Path array for augmenting paths
    path = torch.zeros(n, dtype=torch.long)

    # Main loop: Iterate through each row to find an assignment
    for cur_row in range(m):
        # Prepare for Augmentation Phase
        # shortest_path_cost[j] stores the cost of the shortest path from cur_row to column j
        shortest_path_cost = inf * torch.ones(n, dtype=C.dtype, device=C.device)
        SR = set()  # Set of rows in the current augmenting path tree
        SC = set()  # Set of columns in the current augmenting path tree

        sink = -1  # Stores the column at the end of the augmenting path
        min_val = 0  # Current minimum value for path extension
        i = cur_row  # Current row being processed in path search

        # Find the shortest augmenting path starting from cur_row
        while sink == -1:
            SR = SR.union({i})  # Add current row to SR
            AC_minus_SC = list(AC.difference(SC))  # Columns not yet in SC
            lowest = inf  # Lowest cost found in this iteration
            j_out = -1  # Column corresponding to the lowest cost

            # Explore edges from row i to columns not in SC
            for j in AC_minus_SC:
                if C[i, j] == inf:  # Skip infinite cost edges
                    continue
                # Calculate reduced cost and update shortest path if a shorter path is found
                reduced_cost = min_val + C[i, j] - u[i] - v[j]
                if reduced_cost < shortest_path_cost[j]:
                    shortest_path_cost[j] = reduced_cost
                    path[j] = i  # Record previous node in path

                # Update lowest cost and j_out
                if shortest_path_cost[j] < lowest or (
                    shortest_path_cost[j] == lowest
                    and row4col[j] == -1  # Prefer unassigned columns
                ):
                    lowest = shortest_path_cost[j]
                    j_out = j

            min_val = lowest  # Update min_val for next iteration
            j = j_out  # Current column being processed

            # If no path can be found (e.g. all remaining costs are inf)
            if min_val == inf:
                return None  # No feasible assignment

            SC = SC.union({j})  # Add current column to SC

            # Check if an unassigned column is reached (sink found)
            if row4col[j] == -1:
                sink = j
            else:
                # Continue search from the row matched with column j
                i = row4col[j].item()

        # Update the Dual Variables (u and v)
        u[cur_row] += min_val
        # Update u for rows in SR (except cur_row)
        for i_sr in SR.difference({cur_row}):
            u[i_sr] += min_val - shortest_path_cost[col4row[i_sr]]
        # Update v for columns in SC
        for j_sc in SC:
            v[j_sc] -= min_val - shortest_path_cost[j_sc]

        # Augment the Previous Solution using the found path
        j = sink
        while True:
            i = path[j].item()  # Get the row from the path
            # Update assignments
            row4col[j] = i
            temp = col4row[i].item()  # Store previous assignment for row i
            col4row[i] = j
            # If we reached the starting row of the augmentation, stop
            if i == cur_row:
                break
            j = temp  # Move to the next part of the path to augment

    # If matrix was transposed, revert the assignments
    if transpose:
        # Sort col4row by its values to get row_ind, and use argsort for col_ind
        return v, u, col4row[col4row.argsort()], col4row.argsort()

    # Return dual variables and assignments
    return u, v, torch.arange(m).to(col4row), col4row


if __name__ == "__main__":
    m, n = 5, 10

    C = torch.randn((m, n))

    row_ind, col_ind = linear_sum_assignment(C.numpy())

    u, v, row4col, col4row = hungarian_matching(C.clone())
