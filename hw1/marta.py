import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Set, Optional


class PuzzleState:
    """Represents a state of the 8-puzzle game."""

    def __init__(self, board: np.ndarray):
        """
        Initialize a puzzle state with a 3x3 board.
        
        Args:
            board: 3x3 numpy array representing the puzzle state
        """
        self.board = board
        self.blank_pos = tuple(np.argwhere(board == 0)[0])

    def get_moves(self) -> List[str]:
        """
        Returns possible moves from current state.
        
        Returns:
            List of valid moves (R, L, U, D)
        """
        i, j = self.blank_pos
        moves = []
        # Right move (blank moves left)
        if j > 0:
            moves.append('R')
        # Left move (blank moves right)
        if j < 2:
            moves.append('L')
        # Up move (blank moves down)
        if i < 2:
            moves.append('U')
        # Down move (blank moves up)
        if i > 0:
            moves.append('D')
        return moves

    def apply_move(self, move: str) -> 'PuzzleState':
        """
        Apply a move to the current state and return a new state.
        Note: moves are from the blank tile's perspective (opposite of the named direction)
        
        Args:
            move: One of 'R', 'L', 'U', 'D'
            
        Returns:
            New PuzzleState after applying the move
        """
        i, j = self.blank_pos
        new_board = self.board.copy()
        if move == 'R':  # Blank moves left
            new_board[i, j], new_board[i, j - 1] = new_board[i, j - 1], new_board[i, j]
        elif move == 'L':  # Blank moves right
            new_board[i, j], new_board[i, j + 1] = new_board[i, j + 1], new_board[i, j]
        elif move == 'U':  # Blank moves down
            new_board[i, j], new_board[i + 1, j] = new_board[i + 1, j], new_board[i, j]
        elif move == 'D':  # Blank moves up
            new_board[i, j], new_board[i - 1, j] = new_board[i - 1, j], new_board[i, j]
        return PuzzleState(new_board)

    def flatten(self) -> Tuple:
        """
        Returns a flattened tuple representation of the board for hashing.
        
        Returns:
            Tuple representation of the board
        """
        return tuple(self.board.flatten())

    def __eq__(self, other):
        return np.array_equal(self.board, other.board)

    def __hash__(self):
        return hash(self.flatten())

    def __str__(self):
        return str(self.board)


def manhattan_distance(state: PuzzleState, goal: PuzzleState) -> int:
    """
    Calculate the sum of Manhattan distances of misplaced tiles.
    
    Args:
        state: Current puzzle state
        goal: Goal puzzle state
        
    Returns:
        Sum of Manhattan distances
    """
    total_distance = 0
    # Map tile values to positions in goal state
    goal_positions = {}
    for i in range(3):
        for j in range(3):
            value = goal.board[i, j]
            if value != 0:
                goal_positions[value] = (i, j)
    # Sum Manhattan distances
    for i in range(3):
        for j in range(3):
            value = state.board[i, j]
            if value != 0 and value in goal_positions:
                gi, gj = goal_positions[value]
                total_distance += abs(i - gi) + abs(j - gj)
    return total_distance


class Agent:
    """Agent that uses Multi-Agent RTA* with attraction to solve 8-puzzle.
       In this version, after selecting the best move the agent updates the 
       heuristic value of its current state with the second-best neighbor value,
       so as to discourage other agents from following the same path.
    """

    def __init__(self, initial_state: PuzzleState, goal_state: PuzzleState, attraction_weight: float = 0.5):
        """
        Initialize an agent with initial and goal states.
        
        Args:
            initial_state: Initial puzzle state
            goal_state: Goal puzzle state
            attraction_weight: Weight for the attraction component
        """
        self.current_state = initial_state
        self.goal_state = goal_state
        self.attraction_weight = attraction_weight
        self.history = []  # Store state and move history
        self.h_table: Dict[Tuple, float] = {}  # Learned heuristic values

    def get_learned_heuristic(self, state: PuzzleState) -> float:
        """
        Get the learned heuristic value for a state.
        If no learned value exists, use the default Manhattan distance.
        """
        return self.h_table.get(state.flatten(), manhattan_distance(state, self.goal_state))

    def effective_heuristic(self, state: PuzzleState, other_agents_states: List[PuzzleState]) -> float:
        """
        Compute effective heuristic by combining learned heuristic and attraction component.
        
        Args:
            state: Puzzle state to evaluate
            other_agents_states: List of states of other agents
            
        Returns:
            Effective heuristic value.
        """
        base_value = self.get_learned_heuristic(state)
        h_attraction = 0.0
        if other_agents_states:
            for other_state in other_agents_states:
                # Compute the number of differing tiles as a measure of dissimilarity
                differences = np.sum(state.board != other_state.board)
                h_attraction += differences
            h_attraction /= len(other_agents_states)
        # Subtract the attraction term to favor diversity
        return base_value - self.attraction_weight * h_attraction

    def choose_move(self, other_agents_states: List[PuzzleState]) -> str:
        """
        Choose the best move based on a 1-step lookahead using effective heuristic.
        Also updates the learned heuristic value for the current state with the second-best cost.
        
        Args:
            other_agents_states: States of other agents.
            
        Returns:
            Best move ('R', 'L', 'U', 'D').
        """
        possible_moves = self.current_state.get_moves()
        neighbors = []
        # Evaluate each move: cost = 1 (step cost) + effective heuristic of neighbor
        for move in possible_moves:
            next_state = self.current_state.apply_move(move)
            cost = 1 + self.effective_heuristic(next_state, other_agents_states)
            neighbors.append((move, cost))

        # Sort moves by cost (lower is better)
        neighbors.sort(key=lambda x: x[1])
        best_move, best_cost = neighbors[0]
        # For multiple-agent RTA*, update with second-best value if available
        if len(neighbors) > 1:
            second_best_cost = neighbors[1][1]
        else:
            second_best_cost = best_cost

        # Update the learned heuristic for the current state
        self.h_table[self.current_state.flatten()] = second_best_cost

        return best_move

    def make_move(self, move: str) -> None:
        """
        Make a move and update agent's state.
        
        Args:
            move: One of 'R', 'L', 'U', 'D'
        """
        self.history.append((self.current_state.flatten(), move))
        self.current_state = self.current_state.apply_move(move)

    def has_reached_goal(self) -> bool:
        """
        Check if the agent has reached the goal state.
        
        Returns:
            True if current state equals goal state.
        """
        return self.current_state == self.goal_state


class MARTASolver:
    """Solver that uses Multiple Agents with Real-Time A* with attraction (MARTA*)."""

    def __init__(self, initial_state: PuzzleState, goal_state: PuzzleState, num_agents: int):
        """
        Initialize the solver with a given number of agents.
        
        Args:
            initial_state: Initial puzzle state.
            goal_state: Goal puzzle state.
            num_agents: Number of agents to use.
        """
        self.agents = [Agent(initial_state, goal_state) for _ in range(num_agents)]
        self.goal_state = goal_state
        self.steps = 0
        self.max_steps = 1000  # Prevent infinite loops

    def solve(self) -> List[Dict]:
        """
        Solve the 8-puzzle using MARTA*.
        
        Returns:
            List of steps with agents' actions.
        """
        solution_steps = []
        while self.steps < self.max_steps:
            self.steps += 1
            step_info = {"step": self.steps, "agents": []}
            # Get current states of all agents
            all_states = [agent.current_state for agent in self.agents]
            for agent_idx, agent in enumerate(self.agents):
                # Get states of other agents for attraction
                other_states = [all_states[i] for i in range(len(all_states)) if i != agent_idx]
                move = agent.choose_move(other_states)
                agent.make_move(move)
                step_info["agents"].append({
                    "agent_id": agent_idx + 1,
                    "move": move,
                    "state": agent.current_state.flatten()
                })
                if agent.has_reached_goal():
                    step_info["goal_reached"] = True
                    step_info["goal_agent"] = agent_idx + 1
                    solution_steps.append(step_info)
                    return solution_steps
            solution_steps.append(step_info)
        return solution_steps

    def write_output(self, output_file: str, solution_steps: List[Dict]) -> None:
        """
        Write solution steps to output file in specified format.
        
        Args:
            output_file: Path to output file.
            solution_steps: Solution steps from solve method.
        """
        with open(output_file, 'w') as f:
            for step in solution_steps:
                f.write(f"Step:{step['step']}\n")
                for agent_info in step["agents"]:
                    agent_id = agent_info["agent_id"]
                    move = agent_info["move"]
                    state = agent_info["state"]
                    state_str = ' '.join(map(str, state))
                    f.write(f"Agent{agent_id}: {move} [{state_str}]\n")
                if "goal_reached" in step:
                    f.write(f"\nAgent{step['goal_agent']} reaches the goal.\n")
                    break
                f.write("\n")


def read_input(input_file: str) -> Tuple[int, PuzzleState, PuzzleState]:
    """
    Read input file and create initial and goal states.
    
    Args:
        input_file: Path to the input file.
        
    Returns:
        Tuple of (num_agents, initial_state, goal_state).
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    num_agents = int(lines[0].strip())
    # Parse initial state
    initial_board = np.zeros((3, 3), dtype=int)
    for i in range(3):
        row = lines[i + 1].strip().split()
        for j in range(3):
            initial_board[i, j] = int(row[j])
    # Parse goal state
    goal_board = np.zeros((3, 3), dtype=int)
    for i in range(3):
        row = lines[i + 4].strip().split()
        for j in range(3):
            goal_board[i, j] = int(row[j])
    return num_agents, PuzzleState(initial_board), PuzzleState(goal_board)


def main():
    """Main function to solve 8-puzzle using MARTA*."""
    input_file = "input.txt"
    output_file = "output.txt"
    num_agents, initial_state, goal_state = read_input(input_file)
    solver = MARTASolver(initial_state, goal_state, num_agents)
    solution_steps = solver.solve()
    solver.write_output(output_file, solution_steps)
    goal_step = next((step for step in solution_steps if "goal_reached" in step), None)
    if goal_step:
        print(f"Puzzle solved in {goal_step['step']} steps by Agent{goal_step['goal_agent']}.")
    else:
        print("Failed to solve puzzle within maximum steps.")


if __name__ == "__main__":
    main()
