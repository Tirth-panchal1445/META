import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class IndianTrafficNavigationEnv(gym.Env):
    """
    Indian Traffic Navigation Environment
    Meta PyTorch OpenEnv Hackathon

    Agent (car) navigates a 10x10 grid road with chaotic Indian traffic.
    Goal: Reach the top center safely and quickly.

    Observation: flattened 10x10 grid
        0 = empty
        1 = agent (your car)
        2 = vehicle (collision = -50, episode ends)
        3 = pothole (penalty = -20, continues)
        4 = pedestrian (collision = -100, episode ends)
        5 = goal

    Actions:
        0 = Up
        1 = Left
        2 = Right
        3 = Stay

    Rewards:
        -0.5  every step (encourages efficiency)
        +1.0  bonus for moving forward (up)
        -20   hit a pothole
        -50   crash into vehicle (done)
        -100  hit a pedestrian (done)
        +100  reached the goal (done)
    """

    EMPTY      = 0
    AGENT      = 1
    VEHICLE    = 2
    POTHOLE    = 3
    PEDESTRIAN = 4
    GOAL       = 5

    def __init__(self, grid_size=10, max_steps=100):
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps

        # Observation space: flattened grid, values 0-5
        self.observation_space = spaces.Box(
            low=0, high=5,
            shape=(grid_size * grid_size,),
            dtype=np.int32
        )

        # Action space: 0=Up, 1=Left, 2=Right, 3=Stay
        self.action_space = spaces.Discrete(4)

        self.current_step = 0
        self.agent_pos    = None
        self.goal_pos     = (0, grid_size // 2)   # top center
        self.obstacles    = []

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Return current grid state as flattened numpy array."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Place goal first so agent renders on top if same cell
        grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL

        # Place obstacles
        for r, c, t in self.obstacles:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r, c] = t

        # Place agent (always on top)
        grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT

        return grid.flatten()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset environment to start of a new episode."""

        # Proper seeding (gymnasium standard)
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        random.seed(int(rng.integers(0, 2**31)))

        self.current_step = 0
        self.agent_pos    = (self.grid_size - 1, self.grid_size // 2)

        # Positions that must stay clear
        forbidden = {self.goal_pos, self.agent_pos}

        # Spawn obstacles
        self.obstacles = []
        counts = {
            self.VEHICLE:    random.randint(8, 15),
            self.POTHOLE:    random.randint(3, 7),
            self.PEDESTRIAN: random.randint(4, 8),
        }

        for obs_type, n in counts.items():
            for _ in range(n):
                r = random.randint(1, self.grid_size - 2)
                c = random.randint(0, self.grid_size - 1)
                if (r, c) not in forbidden:
                    self.obstacles.append((r, c, obs_type))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        """
        Take one action and return (obs, reward, done, truncated, info).

        action: int  0=Up  1=Left  2=Right  3=Stay
        """
        self.current_step += 1

        prev_row, _ = self.agent_pos
        row, col    = self.agent_pos

        # Move agent
        if   action == 0: row = max(0, row - 1)                        # Up
        elif action == 1: col = max(0, col - 1)                        # Left
        elif action == 2: col = min(self.grid_size - 1, col + 1)       # Right
        # action == 3: Stay — no movement

        self.agent_pos = (row, col)

        reward    = -0.5   # small step cost
        done      = False
        truncated = False

        # Check collisions
        for r, c, t in self.obstacles:
            if row == r and col == c:
                if t == self.VEHICLE:
                    reward -= 50
                    done    = True
                elif t == self.POTHOLE:
                    reward -= 20
                elif t == self.PEDESTRIAN:
                    reward -= 100
                    done    = True
                break

        # Check goal
        if self.agent_pos == self.goal_pos:
            reward += 100
            done    = True

        # Forward movement bonus
        if action == 0 and row < prev_row:
            reward += 1.0

        # Max steps exceeded
        if self.current_step >= self.max_steps:
            truncated = True

        obs  = self._get_obs()
        info = {
            "step":      self.current_step,
            "agent_pos": self.agent_pos,
            "goal_pos":  self.goal_pos,
        }

        return obs, reward, done, truncated, info

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        """Print a simple ASCII grid to the terminal."""
        SYMBOLS = {
            self.EMPTY:      ".",
            self.AGENT:      "C",
            self.VEHICLE:    "V",
            self.POTHOLE:    "O",
            self.PEDESTRIAN: "P",
            self.GOAL:       "G",
        }

        grid = [[SYMBOLS[self.EMPTY]] * self.grid_size
                for _ in range(self.grid_size)]

        grid[self.goal_pos[0]][self.goal_pos[1]] = SYMBOLS[self.GOAL]

        for r, c, t in self.obstacles:
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                grid[r][c] = SYMBOLS[t]

        # Agent always on top
        grid[self.agent_pos[0]][self.agent_pos[1]] = SYMBOLS[self.AGENT]

        border = "+" + "-" * (self.grid_size * 2 - 1) + "+"
        print(border)
        for row in grid:
            print("|" + " ".join(row) + "|")
        print(border)
        print(f"Step: {self.current_step} | Pos: {self.agent_pos} | Goal: {self.goal_pos}")
        print("C=You  G=Goal  V=Vehicle  O=Pothole  P=Pedestrian")

    # ------------------------------------------------------------------
    def close(self):
        pass
