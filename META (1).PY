import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class IndianTrafficNavigationEnv(gym.Env):
    """
    Indian Traffic Navigation Environment for Meta PyTorch OpenEnv Hackathon
    Agent (car) navigates a grid road with chaotic Indian traffic elements.
    Goal: Reach the top safely and quickly.
    """
    
    def __init__(self, grid_size=10, max_steps=100):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Observation: flattened grid (0=empty, 1=agent, 2=vehicle, 3=pothole, 4=pedestrian)
        self.observation_space = spaces.Box(
            low=0, high=4, shape=(grid_size * grid_size,), dtype=np.int32
        )
        
        # Actions: 0=Up, 1=Left, 2=Right, 3=Stay (simple discrete)
        self.action_space = spaces.Discrete(4)
        
        self.current_step = 0
        self.agent_pos = None
        self.goal_pos = (0, grid_size//2)  # Top center
        self.obstacles = []  # vehicles, potholes, pedestrians
        
    def _get_obs(self):
        """Return current grid as flattened array"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place agent
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        
        # Place goal (visible as special marker, say 5)
        grid[self.goal_pos[0], self.goal_pos[1]] = 5
        
        # Place obstacles
        for obs in self.obstacles:
            if 0 <= obs[0] < self.grid_size and 0 <= obs[1] < self.grid_size:
                grid[obs[0], obs[1]] = obs[2]  # type: 2=vehicle, 3=pothole, 4=pedestrian
        
        return grid.flatten()
    
    def reset(self, seed=None, options=None):
        """Reset environment to start of episode"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.current_step = 0
        
        # Agent starts at bottom center
        self.agent_pos = (self.grid_size - 1, self.grid_size // 2)
        
        # Generate random obstacles (Indian traffic feel)
        self.obstacles = []
        num_vehicles = random.randint(8, 15)
        num_potholes = random.randint(3, 7)
        num_pedestrians = random.randint(4, 8)
        
        # Vehicles (moving obstacles - type 2)
        for _ in range(num_vehicles):
            x = random.randint(1, self.grid_size-2)
            y = random.randint(0, self.grid_size-1)
            self.obstacles.append((x, y, 2))
        
        # Potholes (static dangerous - type 3)
        for _ in range(num_potholes):
            x = random.randint(1, self.grid_size-2)
            y = random.randint(0, self.grid_size-1)
            self.obstacles.append((x, y, 3))
        
        # Pedestrians (slow moving - type 4)
        for _ in range(num_pedestrians):
            x = random.randint(1, self.grid_size-2)
            y = random.randint(0, self.grid_size-1)
            self.obstacles.append((x, y, 4))
        
        # Ensure goal is not blocked
        self.obstacles = [obs for obs in self.obstacles if obs[:2] != self.goal_pos]
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Take action and return next observation, reward, done, info"""
        self.current_step += 1
        reward = 0
        done = False
        truncated = False
        
        # Move agent based on action
        row, col = self.agent_pos
        
        if action == 0:   # Up (forward - main direction in India roads)
            row = max(0, row - 1)
        elif action == 1: # Left
            col = max(0, col - 1)
        elif action == 2: # Right
            col = min(self.grid_size - 1, col + 1)
        # action 3: Stay (useful for waiting at "red light" or avoiding collision)
        
        self.agent_pos = (row, col)
        
        # Small negative reward per step (encourage efficiency)
        reward -= 0.5
        
        # Check collision with obstacles
        for obs in self.obstacles:
            if (row, col) == obs[:2]:
                if obs[2] == 2:   # Vehicle collision - big penalty (real accident)
                    reward -= 50
                    done = True
                elif obs[2] == 3: # Pothole
                    reward -= 20
                elif obs[2] == 4: # Pedestrian (hit person - very bad)
                    reward -= 100
                    done = True
                break
        
        # Reached goal?
        if self.agent_pos == self.goal_pos:
            reward += 100  # Big success reward
            done = True
        
        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
        
        # Optional: small bonus for moving forward
        if action == 0 and row < self.agent_pos[0]:  # Wait, row decreases when going up
            reward += 1
        
        obs = self._get_obs()
        info = {"step": self.current_step}
        
        return obs, reward, done, truncated, info
    
    def render(self, mode="human"):
        """Simple text render for debugging"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = "."
        
        grid[self.agent_pos[0], self.agent_pos[1]] = "🚗"
        grid[self.goal_pos[0], self.goal_pos[1]] = "🏁"
        
        for obs in self.obstacles:
            if obs[2] == 2:
                grid[obs[0], obs[1]] = "🚙"  # Vehicle
            elif obs[2] == 3:
                grid[obs[0], obs[1]] = "🕳️"  # Pothole
            elif obs[2] == 4:
                grid[obs[0], obs[1]] = "🚶"  # Pedestrian
        
        print("\n".join(" ".join(row) for row in grid))
        print(f"Step: {self.current_step} | Pos: {self.agent_pos}")

      