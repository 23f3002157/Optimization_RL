# Deep Q-Learning for Linear Optimization

## Overview
This project demonstrates the application of a **Deep Q-Learning (DQN)** agent to solve a constrained linear optimization problem in three dimensions.  
The implementation is provided in the Jupyter notebook [`three_var_solved.ipynb`](./three_var_solved.ipynb) using Python.  

The DQN agent learns to **maximize an objective function while adhering to constraints**, leveraging reinforcement learning techniques. The notebook includes code for:

- Custom 3D environment  
- DQN model and replay buffer  
- Training loop  
- Visualization of results  

The optimization problem solved is illustrated below:

<p align="center">
  <img src="./images/three_v.png" alt="3D Linear Optimization Problem" width="400"/>
</p>

---

## Problem Description

We want to maximize the objective:

```math
z = 60x_1 + 30x_2 + 20x_3
``` 
subject to the following constraints:

```math
8x_1 + 6x_2 + x_3 <= 48
```

```math
4x_1 + 2x_2 + 1.5x_3 <= 20
```

```math
2x_1 + 1.5x_2 + 0.5x_3 <= 8
```

```math
x_2 <= 5
```

```math
x_1, x_2, x_3 >= 0
```

## Setup

### 1. Clone the Repository:

```
git clone https://github.com/23f3002157/Optimization_RL.git
cd Three_Variables
```

### 2. Install Dependencies: 
Ensure Python 3.12.8 or later is installed, then run:

```
pip install -r requirements.txt
```

### 3. Hardware Requirements:

```
The code is configured to use the MPS backend (torch.device("mps")) for Apple Silicon GPUs. For non-MPS systems, modify the device to "cpu" in the notebook (Cell 2). 
A standard CPU is sufficient for running the notebook, though training may be slower.
```

### 4. Usage

```
1. Open the Notebook: Launch Jupyter Notebook or JupyterLab and open three_var_solved.ipynb:

2. jupyter notebook three_var_solved.ipynb

3. Run the Cells:

    3.1 Execute all cells sequentially to:
    3.2 Load libraries and set random seeds (Cells 1-4).
    3.3 Define the DQN model and replay buffer (Cells 5-6).
    3.4 Define the 3D optimization problem and environment (Cells 7-8).
    3.5 Initialize the environment.
    3.6 Set up the DQN model and optimizer (Cell 10).
    3.7 Train the DQN agent (Cell 11).
    3.8 Visualize training metrics and final states (Cells 12-14).

4. Training:

    Run Cell 11 to train the DQN agent for 2000 episodes. Training logs will print every 100 episodes, showing average reward, epsilon, final state, average objective, and average constraint violation.

5. Visualization:
    Cell 12 plots smoothed rewards vs. episodes. Cell 13 plots average constraint violations vs. episodes. Cell 14 plots the final states in 3D space, with the feasible region and optimal point highlighted, as depicted in images/three_v.jpeg.

6. Results

    The DQN agent learns to converge to the optimal point (2, 0, 8), achieving an objective value of 280. 

7. Notes

    Hyperparameters: Tuned for stability (e.g., LR=5e-4, EPSILON_DECAY=20000, PENALTY=100.0). Adjust in Cell 3 or 11 if needed.
```