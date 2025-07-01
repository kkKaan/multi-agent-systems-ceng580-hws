import heapq, math, sys


# --------------------
# 1. Parser Functions
# --------------------
def load_map(map_filename):
    with open(map_filename) as f:
        lines = [line.strip() for line in f if line.strip()]
    # Skip header "type ... height ... width ..." lines
    # Map data starts after a line "map"
    idx = lines.index("map")
    grid = [list(lines[i]) for i in range(idx + 1, len(lines))]
    H = len(grid)
    W = len(grid[0])
    obstacles = [[(grid[y][x] != '.' and grid[y][x] != 'G') for x in range(W)] for y in range(H)]
    return obstacles, H, W, grid


def load_scen(scen_filename):
    agents = []
    with open(scen_filename) as f:
        for line in f:
            if line.startswith("version"):
                continue
            fields = line.split()
            if len(fields) >= 7 and fields[0] == '0':  # bucket 0
                sx, sy = int(fields[4]), int(fields[5])
                gx, gy = int(fields[6]), int(fields[7])
                agents.append(((sx, sy), (gx, gy)))
    return agents


# -------------------------
# 2. Low-Level Round-Trip A*
# -------------------------
def round_trip(start, goal, obs, cons):
    W, H = len(obs[0]), len(obs)
    sx, sy = start
    gx, gy = goal

    # If start and goal are the same location
    if (sx, sy) == (gx, gy):
        return [(sx, sy)], 0  # Return a path with just one point and cost 0

    def neigh(x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not obs[ny][nx]:
                yield nx, ny

    def heuristic(x, y, stage):
        h0 = math.hypot(x - gx, y - gy) + math.hypot(gx - sx, gy - sy)
        h1 = math.hypot(x - sx, y - sy)
        return h0 if stage == 0 else h1

    open_, closed = [], set()
    start_h = heuristic(sx, sy, 0)
    heapq.heappush(open_, (start_h, 0, sx, sy, 0, 0, None))  # (f,g,x,y,stage,t,prev)

    while open_:
        f, g, x, y, stage, t, prev = heapq.heappop(open_)
        if (x, y, stage, t) in closed:
            continue
        closed.add((x, y, stage, t))

        if stage == 1 and (x, y) == (sx, sy):
            # reconstruct path
            path = []
            node = (x, y, stage, t, prev)
            while node:
                x0, y0, st0, t0, p0 = node
                path.append((x0, y0))
                node = p0
            return path[::-1], g

        # stage switch at goal
        next_stage = 1 if (stage == 0 and (x, y) == (gx, gy)) else stage

        # wait - check vertex constraints properly
        vertex_constrained = any(c == (x, y, t + 1) for c in cons if isinstance(c, tuple) and len(c) == 3)
        if not vertex_constrained:
            h = heuristic(x, y, next_stage)
            heapq.heappush(open_, (g + 1 + h, g + 1, x, y, next_stage, t + 1, (x, y, stage, t, prev)))

        # moves - check both vertex and edge constraints properly
        for nx, ny in neigh(x, y):
            vertex_constrained = any(
                c == (nx, ny, t + 1) for c in cons if isinstance(c, tuple) and len(c) == 3)
            edge_constrained = any(
                c == (x, y, nx, ny, t) for c in cons if isinstance(c, tuple) and len(c) == 5)

            if vertex_constrained or edge_constrained:
                continue

            h = heuristic(nx, ny, next_stage)
            heapq.heappush(open_, (g + 1 + h, g + 1, nx, ny, next_stage, t + 1, (x, y, stage, t, prev)))

    # If no path found
    return None, float('inf')


# ------------------------
# 3. Conflict Detection
# ------------------------
def detect_conflict(paths):
    # Create space-time occupation map for precise tracking
    occupations = {}  # (x, y, t) -> list of agents at this position at time t
    edge_moves = {}  # (x1, y1, x2, y2, t) -> list of agents making this move at time t

    # Fill occupation and movement maps with full agent timelines
    T = max(len(p) for p in paths)
    for i, path in enumerate(paths):
        for t in range(T):
            # Get current position (last position if path ended)
            curr_pos = path[min(t, len(path) - 1)]
            pos_key = (*curr_pos, t)

            # Record in occupations map
            if pos_key not in occupations:
                occupations[pos_key] = []
            occupations[pos_key].append(i)

            # Record edge movements (only if agent is still moving)
            if t + 1 < len(path):
                next_pos = path[t + 1]
                move_key = (*curr_pos, *next_pos, t)
                if move_key not in edge_moves:
                    edge_moves[move_key] = []
                edge_moves[move_key].append(i)

    # Check vertex conflicts first (higher priority)
    for (x, y, t), agents in occupations.items():
        if len(agents) > 1:
            return (agents[0], agents[1], ((x, y)), t, 'vertex')

    # Check edge conflicts
    for (x1, y1, x2, y2, t), agents_moving in edge_moves.items():
        # Check if any agents are moving in the opposite direction
        reverse_key = (x2, y2, x1, y1, t)
        if reverse_key in edge_moves:
            return (agents_moving[0], edge_moves[reverse_key][0], (((x1, y1), (x2, y2))), t, 'edge')

    return None


# --------------------------
# 4. High-Level CBS Search
# --------------------------
class CTNode:

    def __init__(self, cons, sol, cost):
        self.cons, self.sol, self.cost = cons, sol, cost

    def __lt__(self, other):
        return self.cost < other.cost


def cbs(obstacles, agents, max_iterations=1000):
    N = len(agents)
    emptyC = {i: set() for i in range(N)}
    # root
    sol, cost = [], 0
    for i, (s, g) in enumerate(agents):
        p, c = round_trip(s, g, obstacles, emptyC[i])
        sol.append(p)
        cost += c
    root = CTNode(emptyC, sol, cost)
    pq = [(cost, root)]

    iterations = 0
    while pq and iterations < max_iterations:
        iterations += 1
        _, node = heapq.heappop(pq)
        conflict = detect_conflict(node.sol)
        if not conflict:
            return node.sol
        i, j, cell, t, typ = conflict

        for ag in (i, j):
            newC = {a: set(node.cons[a]) for a in node.cons}
            if typ == 'vertex':
                x, y = cell
                newC[ag].add((x, y, t))
            else:
                # Properly format edge constraints
                (x1, y1), (x2, y2) = cell[0], cell[1]
                newC[ag].add((x1, y1, x2, y2, t))

            newSol = list(node.sol)
            p, c = round_trip(agents[ag][0], agents[ag][1], obstacles, newC[ag])
            if p is None:
                continue
            newSol[ag] = p
            newCost = sum(len(p_) - 1 for p_ in newSol)
            heapq.heappush(pq, (newCost, CTNode(newC, newSol, newCost)))

    if iterations >= max_iterations:
        # print(f"Warning: CBS reached maximum iterations ({max_iterations})")

        # Return best solution found so far if any exists in queue
        if pq:
            best_cost = float('inf')
            best_node = None
            for cost, node in pq:
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
            if best_node:
                return best_node.sol

    return None


# ------------------------------
# 5. Main & Output Writer
# ------------------------------
def main():
    if len(sys.argv) != 3:
        print("Usage: main.py <mapName.map> <scenName.map.scen>")
        sys.exit(1)
    map_file, scen_file = sys.argv[1], sys.argv[2]
    obstacles, H, W, grid = load_map(map_file)
    agents = load_scen(scen_file)

    # # Print agent starting positions to terminal
    # for k, ((sx, sy), _) in enumerate(agents):
    #     map_char = grid[sy][sx]
    #     print(f"Agent{str(k+1).zfill(2)} starting position: ({sx},{sy}) [{map_char}]")

    # Use modified CBS with iteration limit
    solution = cbs(obstacles, agents, max_iterations=10000)

    if solution:
        with open("output.txt", "w") as out:
            for k, (_, g) in enumerate(agents):
                path = solution[k]
                cost = len(path) - 1
                coords = []
                for x, y in path:
                    coords.append(f"[{x},{y}]" if (x, y) == g else f"({x},{y})")
                out.write(f"Agent{str(k+1).zfill(2)}, PathCost : {cost}, Path : {' '.join(coords)}\n")
    # else:
    #     print("No solution found within iteration limit.")


# I designed it to be run from the command line with two arguments: map file and scenario file.
if __name__ == "__main__":
    main()
