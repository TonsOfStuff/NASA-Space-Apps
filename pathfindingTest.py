import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def find_rover_route(elevation_crop, start_rc, target_rc, pixel_m, max_slope_deg=15.0):
  rows, cols = elevation_crop.shape
  max_tan = np.tan(np.radians(max_slope_deg))

  # 1. Build a sparse graph of valid directional moves (8-connected neighbors)
  edges_from, edges_to, weights = [], [], []
  directions = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

  r_idx, c_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
  node_ids = r_idx * cols + c_idx

  for dr, dc, dist_mult in directions:
    # Slice overlapping valid neighbor grids using vectorized NumPy
    r1 = slice(max(0, -dr), min(rows, rows - dr))
    c1 = slice(max(0, -dc), min(cols, cols - dc))
    r2 = slice(max(0, dr), min(rows, rows + dr))
    c2 = slice(max(0, dc), min(cols, cols + dc))

    dz = np.abs(elevation_crop[r2, c2] - elevation_crop[r1, c1])
    dist = dist_mult * pixel_m
    slope = dz / dist

    # Keep only moves within the rover's slope limit
    valid = slope <= max_tan
    u = node_ids[r1, c1][valid]
    v = node_ids[r2, c2][valid]

    # Cost function: penalizes slope while minimizing total distance
    cost = dist * (1.0 + (slope[valid] / max_tan) ** 2)

    edges_from.append(u)
    edges_to.append(v)
    weights.append(cost)

  graph = csr_matrix((np.concatenate(weights), (np.concatenate(edges_from), np.concatenate(edges_to))), shape=(rows * cols, rows * cols))

  start_node = start_rc[0] * cols + start_rc[1]
  target_node = target_rc[0] * cols + target_rc[1]

  # 2. Run fast C-compiled Dijkstra from start
  dist_matrix, predecessors = dijkstra(graph, directed=True, indices=start_node, return_predecessors=True)

  # 3. Check if target is reachable; if not, pick closest reachable parking spot
  if np.isinf(dist_matrix[target_node]):
    reachable = np.where(~np.isinf(dist_matrix))[0]
    rem_r, rem_c = np.divmod(reachable, cols)
    dist_to_goal = np.hypot(rem_r - target_rc[0], rem_c - target_rc[1])
    best_node = reachable[np.argmin(dist_to_goal)]
    status = "PARKED_AT_DEAD_END"
  else:
    best_node = target_node
    status = "REACHED_TARGET"

  # 4. Reconstruct path backwards
  path = []
  curr = best_node
  while curr != -9999 and curr != start_node:
    path.append(divmod(curr, cols))
    curr = predecessors[curr]
  path.append(start_rc)
  return path[::-1], status
