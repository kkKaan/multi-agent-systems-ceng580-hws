"""
MARTA* Multi-Agent Real-Time Search Implementation for the 8-Puzzle

Each agent performs a one-step lookahead (RTA* style) from the current state.
They share a global heuristic table (h_G) while maintaining their own local updates (h_L)
for visited states. In the choice step an attraction mechanism is applied: from the candidate
set of moves with minimum f = 1 + h(child), the agent computes an isolation measure (max Manhattan
distance between candidate's blank and other agents' blanks). In addition, agents now try to select
different moves from each other if alternative moves exist.

Input:
  - First line: number of agents (n)
  - Next 3 lines: initial 3x3 board (row-wise, blank is denoted by 0)
  - Next 3 lines: goal 3x3 board
Output:
  - For each step, each agent's move and resulting state (flattened)
  - When one agent reaches the goal, output the winning agent and stop.
  
Author: Kaan Karacanta
"""

import random
from typing import List, Tuple, Dict, Optional

# Global dictionaries for shared heuristic estimates
global_h: Dict[Tuple[int, ...], int] = {}

# Type alias for state representation (flattened tuple of 9 ints)
State = Tuple[int, ...]

# Goal positions for Manhattan heuristic (computed from goal state)
goal_positions: Dict[int, Tuple[int, int]] = {}

# Attraction range parameter (tunable)
ATTRACTION_RANGE = 2  # G parameter


def read_input(filename: str) -> Tuple[int, State, State]:
    """Reads input file and returns number of agents, initial state and goal state."""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    num_agents = int(lines[0])
    init_board = []
    goal_board = []
    # next 3 lines: initial state
    for i in range(1, 4):
        init_board.extend([int(x) for x in lines[i].split()])
    # next 3 lines: goal state
    for i in range(4, 7):
        goal_board.extend([int(x) for x in lines[i].split()])
    return num_agents, tuple(init_board), tuple(goal_board)


def write_output(filename: str, output_lines: List[str]) -> None:
    """Writes the list of output lines to output file."""
    with open(filename, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')


def manhattan_heuristic(state: State) -> int:
    """Computes the sum of Manhattan distances for all tiles (ignoring blank) given the global goal_positions."""
    total = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        current_row, current_col = divmod(index, 3)
        goal_row, goal_col = goal_positions[tile]
        total += abs(current_row - goal_row) + abs(current_col - goal_col)
    return total


def get_blank_pos(state: State) -> Tuple[int, int]:
    """Returns (row, col) of the blank (0) in the given state."""
    idx = state.index(0)
    return divmod(idx, 3)


def generate_neighbors(state: State) -> List[Tuple[State, str]]:
    """
    Given a state, returns list of (neighbor_state, move_letter) pairs.
    The move letter indicates the direction of the tile that moves into the blank:
      - 'D': tile from above moves down
      - 'U': tile from below moves up
      - 'R': tile from left moves right
      - 'L': tile from right moves left
    """
    neighbors = []
    blank_row, blank_col = get_blank_pos(state)
    state_list = list(state)

    def swap_and_create(new_r: int, new_c: int, move_letter: str) -> State:
        new_state = list(state)
        blank_index = blank_row * 3 + blank_col
        neighbor_index = new_r * 3 + new_c
        # Swap positions: the neighbor tile moves into blank.
        new_state[blank_index], new_state[neighbor_index] = new_state[neighbor_index], new_state[blank_index]
        return tuple(new_state)

    if blank_row > 0:
        neighbors.append((swap_and_create(blank_row - 1, blank_col, 'D'), 'D'))
    if blank_row < 2:
        neighbors.append((swap_and_create(blank_row + 1, blank_col, 'U'), 'U'))
    if blank_col > 0:
        neighbors.append((swap_and_create(blank_row, blank_col - 1, 'R'), 'R'))
    if blank_col < 2:
        neighbors.append((swap_and_create(blank_row, blank_col + 1, 'L'), 'L'))
    return neighbors


def state_to_vector(state: State) -> str:
    """Converts the state tuple into a space separated vector string."""
    return ' '.join(str(x) for x in state)


class Agent:
    """Class representing an agent in the multi-agent MARTA* search."""

    def __init__(self, agent_id: int, init_state: State):
        self.agent_id = agent_id
        self.current_state: State = init_state
        # Local heuristic updates for visited states: state -> value
        self.h_local: Dict[State, int] = {}
        # Keep track of visited states
        self.visited = set()
        # Record the last move taken (letter)
        self.last_move: Optional[str] = None
        # Give each agent a unique preference for move directions
        self.direction_preferences = {
            1: ['U', 'L', 'D', 'R'],  # Agent 1 prefers up, then left
            2: ['R', 'D', 'L', 'U'],
            3: ['L', 'U', 'R', 'D']
        }.get(agent_id, list("UDLR"))  # Default if more than 3 agents

    def get_heuristic(self, state: State) -> int:
        """Returns the heuristic value to use for a given state."""
        if state in self.visited:
            return self.h_local.get(state, global_h.get(state, manhattan_heuristic(state)))
        else:
            if state not in global_h:
                global_h[state] = manhattan_heuristic(state)
            return global_h[state]

    def update_heuristics(self, current_state: State, best_f: int, second_best_f: Optional[int]) -> None:
        """Updates the global and local heuristic estimates for the given state."""
        global_h[current_state] = best_f
        if second_best_f is None:
            self.h_local[current_state] = float('inf')
        else:
            self.h_local[current_state] = second_best_f

    def mark_visited(self, state: State) -> None:
        """Mark a state as visited."""
        self.visited.add(state)

    def choose_move(self, agents_current: List[State],
                    taken_candidates: List[Tuple[State, str]]) -> Tuple[State, str]:
        """
        Performs one-step lookahead from the current state.
        Returns the chosen next state and the move letter.
        Incorporates an attraction mechanism:
        - First, restrict to moves that are not already taken by other agents (if possible).
        - Compute f = 1 + h(child) for each neighbor.
        - From those with minimum f, compute an isolation measure:
                isolation = max(ManhattanDistance(candidate_blank, other_agent_blank))
            over the other agents' current states.
        - If at least one candidate has an isolation measure <= ATTRACTION_RANGE,
            choose randomly among those; otherwise, choose the candidate with minimum isolation.
        """
        neighbors = generate_neighbors(self.current_state)
        if not neighbors:
            return self.current_state, ''

        # Get all available move letters from neighbors
        available_moves = [move for _, move in neighbors]
        # Moves already taken by other agents in this step
        taken_moves = [move for _, move in taken_candidates]
        # Prefer moves that have not been taken by other agents
        unique_moves = [move for move in available_moves if move not in taken_moves]
        if unique_moves:
            preferred_neighbors = [(state, move) for state, move in neighbors if move in unique_moves]
        else:
            preferred_neighbors = neighbors

        # Compute f-values for the preferred neighbors
        candidate_list = []
        for n_state, move_letter in preferred_neighbors:
            h_val = self.get_heuristic(n_state)
            f_val = 1 + h_val
            try:
                # Apply preference bonus based on the agent's direction preference, to see different moves I added
                preference_bonus = 0.1 * (4 - self.direction_preferences.index(move_letter)) / 10
                f_val -= preference_bonus
            except ValueError:
                pass
            candidate_list.append((n_state, move_letter, f_val))

        # Select candidate moves with minimum f-value
        min_f = min(f for (_, _, f) in candidate_list)
        candidate_moves = [(state, move, f) for (state, move, f) in candidate_list if f == min_f]

        # Incorporate the attraction mechanism:
        # For each candidate, compute its isolation measure based on its blank position relative
        # to the blank positions of the other agents (using Manhattan distance).
        candidate_isolations = []
        for cand_state, move_letter, f_val in candidate_moves:
            cand_blank = get_blank_pos(cand_state)
            distances = []
            for other_state in agents_current:
                # Skip self's current state
                if other_state == self.current_state:
                    continue
                other_blank = get_blank_pos(other_state)
                d = abs(cand_blank[0] - other_blank[0]) + abs(cand_blank[1] - other_blank[1])
                distances.append(d)
            # If no other agents, set isolation to infinity
            isolation = max(distances) if distances else float('inf')
            candidate_isolations.append((cand_state, move_letter, f_val, isolation))

        # Determine candidates based on the attraction threshold (ATTRACTION_RANGE)
        # If at least one candidate has isolation measure <= ATTRACTION_RANGE,
        # choose randomly among them; otherwise, choose the candidate with the minimum isolation.
        candidates_in_range = [
            (s, m, f, iso) for (s, m, f, iso) in candidate_isolations if iso <= ATTRACTION_RANGE
        ]
        if candidates_in_range:
            chosen_candidate = random.choice(candidates_in_range)
        else:
            min_iso = min(iso for (_, _, _, iso) in candidate_isolations)
            best_candidates = [(s, m, f, iso) for (s, m, f, iso) in candidate_isolations if iso == min_iso]
            chosen_candidate = random.choice(best_candidates)
        chosen_state, chosen_move, chosen_f, chosen_iso = chosen_candidate

        # Compute heuristic updates for all neighbors (regardless of the attraction mechanism)
        all_f_values = []
        for n_state, move_letter in neighbors:
            h_val = self.get_heuristic(n_state)
            f_val = 1 + h_val
            all_f_values.append((n_state, move_letter, f_val))
        sorted_f = sorted(all_f_values, key=lambda x: x[2])
        best_f = sorted_f[0][2]
        second_best_f = sorted_f[1][2] if len(sorted_f) > 1 else None

        self.update_heuristics(self.current_state, best_f, second_best_f)
        self.mark_visited(self.current_state)
        self.last_move = chosen_move
        return chosen_state, chosen_move


def marta_star(num_agents: int, init_state: State, goal_state: State) -> List[str]:
    """
    Runs the multi-agent MARTA* search on the 8-puzzle.
    Returns the list of output lines.
    """
    output_lines = []
    for index, tile in enumerate(goal_state):
        row, col = divmod(index, 3)
        goal_positions[tile] = (row, col)
    if init_state not in global_h:
        global_h[init_state] = manhattan_heuristic(init_state)

    agents = [Agent(agent_id=i + 1, init_state=init_state) for i in range(num_agents)]
    step = 0
    reached_agent_id = None

    while True:
        step += 1
        output_lines.append(f"Step : {step} ")
        agents_current_states = [agent.current_state for agent in agents]
        new_states = []
        moves = []
        # This list keeps track of (state, move) chosen by previous agents in this step.
        taken_candidates: List[Tuple[State, str]] = []
        for agent in agents:
            if agent.current_state == goal_state:
                new_states.append(agent.current_state)
                moves.append('')
                continue
            next_state, move_letter = agent.choose_move(agents_current_states, taken_candidates)
            # Record the candidate move to avoid duplication by subsequent agents if alternatives exist.
            taken_candidates.append((next_state, move_letter))
            new_states.append(next_state)
            moves.append(move_letter)
        for i, agent in enumerate(agents):
            agent.current_state = new_states[i]
            line = f"Agent{i+1}: {moves[i]} [{state_to_vector(agent.current_state)}]"
            output_lines.append(line)
        output_lines.append("")
        for agent in agents:
            if agent.current_state == goal_state:
                reached_agent_id = agent.agent_id
                break
        if reached_agent_id is not None:
            output_lines.append(f"Agent{reached_agent_id} reaches the goal.")
            break
        if step > 1000:
            output_lines.append("No solution found within 1000 steps.")
            break
    return output_lines


def main() -> None:
    """Main function: read input, run MARTA*, and write output."""
    input_filename = "input.txt"
    output_filename = "output.txt"
    num_agents, init_state, goal_state = read_input(input_filename)
    output_lines = marta_star(num_agents, init_state, goal_state)
    write_output(output_filename, output_lines)


if __name__ == "__main__":
    main()
