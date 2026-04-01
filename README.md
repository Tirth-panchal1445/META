---
title: Indian Traffic Navigation Environment
emoji: 🚗
colorFrom: yellow
colorTo: red
sdk: docker
sdk_version: "3.10"
python_version: "3.10"
app_file: inference.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🚗 Indian Traffic Navigation Environment

A custom [Gymnasium](https://gymnasium.farama.org/) reinforcement learning environment simulating chaotic Indian traffic conditions, served via a Flask REST API.

Built for the **Meta PyTorch OpenEnv Hackathon**.

---

## 📌 Overview

An agent (car) navigates a **10×10 grid** filled with vehicles, potholes, and pedestrians, starting from the bottom center and aiming to reach the **top center (goal)**. The environment rewards efficient, safe navigation and penalizes collisions and reckless movement.

---

## 🗺️ Grid Legend

| Symbol | Value | Description |
|--------|-------|-------------|
| `.` | 0 | Empty cell |
| `C` | 1 | Agent (your car) |
| `V` | 2 | Vehicle — collision ends episode (`-50`) |
| `O` | 3 | Pothole — penalty, continues (`-20`) |
| `P` | 4 | Pedestrian — collision ends episode (`-100`) |
| `G` | 5 | Goal — reach to win (`+100`) |

---

## 🎮 Action Space

| Action | Value |
|--------|-------|
| Up | `0` |
| Left | `1` |
| Right | `2` |
| Stay | `3` |

---

## 🏆 Reward Structure

| Event | Reward |
|-------|--------|
| Every step | `-0.5` |
| Moving forward (Up) | `+1.0` |
| Hit a pothole | `-20` |
| Crash into vehicle | `-50` + episode ends |
| Hit a pedestrian | `-100` + episode ends |
| Reached goal | `+100` + episode ends |

---

## 📡 REST API

The environment is exposed as a Flask HTTP server running on port `7860`.

### `POST /reset`

Resets the environment and returns the initial observation.

**Response:**
```json
{
  "observation": [0, 0, 1, ...],
  "info": {}
}
```

### `POST /step`

Takes one step in the environment.

**Request body:**
```json
{
  "action": 0
}
```

**Response:**
```json
{
  "observation": [0, 0, 1, ...],
  "reward": -0.5,
  "done": false,
  "truncated": false,
  "info": {
    "step": 1,
    "agent_pos": [8, 5],
    "goal_pos": [0, 5]
  }
}
```

---

## 🐳 Running with Docker

### Build the image

```bash
docker build -t indian-traffic-env .
```

### Run the container

```bash
docker run -p 7860:7860 indian-traffic-env
```

The API will be available at `http://localhost:7860`.

---

## 🛠️ Running Locally (without Docker)

### Install dependencies

```bash
pip install gymnasium numpy flask
```

### Start the server

```bash
python inference.py
```

---

## 🔬 Using the Environment Directly (Python)

```python
from env import IndianTrafficNavigationEnv

env = IndianTrafficNavigationEnv()
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # Replace with your agent
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break

env.close()
```

---

## 📁 Project Structure

```
.
├── env.py          # Gymnasium environment definition
├── inference.py    # Flask API server
├── Dockerfile      # Container setup
└── README.md
```

---

## ⚙️ Environment Details

| Parameter | Value |
|-----------|-------|
| Grid size | 10×10 |
| Max steps per episode | 100 |
| Start position | Bottom center `(9, 5)` |
| Goal position | Top center `(0, 5)` |
| Observation space | `Box(0, 5, shape=(100,), dtype=int32)` |
| Action space | `Discrete(4)` |

---

## 📦 Dependencies

- Python 3.10
- [Gymnasium](https://gymnasium.farama.org/)
- [NumPy](https://numpy.org/)
- [Flask](https://flask.palletsprojects.com/)
