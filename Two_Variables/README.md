Deep Q-Learning for Linear Optimization

Overview

This project demonstrates the application of a Deep Q-Learning (DQN) agent to solve a constrained linear optimization problem. The implementation is provided in a Jupyter notebook (two_var_solved.ipynb) using Python. The DQN agent learns to maximize an objective function while adhering to given constraints, leveraging reinforcement learning techniques. The notebook includes code for the environment, DQN model, training loop, and visualization of results.

The optimization problem solved in this project is depicted in the image images/two_v.jpeg.

Problem Description

The goal is to maximize the objective function ( z = 4x + 3y ) subject to the following constraints:





( x + y \leq 40 )



( 2x + y \leq 60 )



( x \geq 0 ), ( y \geq 0 )

The optimal solution is at ( (x, y) = (20, 20) ), where ( z = 140 ). The DQN agent navigates a 2D state space ([x, y] \in [0, 100]^2), taking actions to adjust ( x ) and ( y ) (up, down, left, right) with a step size of 1. The environment provides rewards based on the objective value when constraints are satisfied, or a penalty for constraint violations.

Dependencies

The following Python packages are required to run the notebook:





torch>=2.0.0: For neural network implementation and training.



matplotlib>=3.8.0: For plotting training metrics and final states.



numpy>=1.26.0: For numerical operations.



gymnasium>=0.29.0: For environment structure (though minimally used).

Install the dependencies using the provided requirements.txt:

pip install -r requirements.txt

Setup





Clone the Repository:

git clone <repository-url>
cd <repository-directory>



Install Dependencies: Ensure Python 3.12.8 or later is installed, then run:

pip install -r requirements.txt



Hardware Requirements:





The code is configured to use the MPS backend (torch.device("mps")) for Apple Silicon GPUs. For non-MPS systems, modify the device to "cpu" in the notebook (Cell 2).



A standard CPU is sufficient for running the notebook, though training may be slower.

Usage





Open the Notebook: Launch Jupyter Notebook or JupyterLab and open two_var_solved.ipynb:

jupyter notebook two_var_solved.ipynb



Run the Cells:





Execute all cells sequentially to:





Load libraries and set random seeds (Cells 1-4).



Define the DQN model and replay buffer (Cells 5-6).



Define the optimization problem and environment (Cells 7-8).



Initialize the environment (Cell 9, note the fix below).



Set up the DQN model and optimizer (Cell 10).



Train the DQN agent (Cell 11).



Visualize training metrics and final states (Cells 12-14).



Fix for Cell 9: The notebook has an error in Cell 9 (env.n_actions is undefined). Replace with:

env = LinearOptimizationEnv()
state, _ = env.reset()
n_obs = len(state)
n_act = env.action_space_size



Training:





Run Cell 11 (a, b, c, d, rew, max_obj = train_dqn()) to train the DQN agent for 2000 episodes.



Training logs will print every 100 episodes, showing average reward, epsilon, final state, average objective, and average constraint violation.



Visualization:





Cell 12 plots smoothed rewards vs. episodes.



Cell 13 plots average constraint violations and objective values vs. episodes.



Cell 14 plots the feasible region, constraints, and final states, with the problem depicted in images/two_v.jpeg.

Results

The DQN agent learns to converge to the optimal point ( (x, y) = (20, 20) ), achieving an objective value ( z \approx 140 ) with zero constraint violations. Key outputs include:





Smoothed Rewards: Plotted in Cell 12, showing cumulative rewards approaching positive values as the agent learns to stay feasible and maximize ( z ).



Constraint Violations and Objectives: Plotted in Cell 13, confirming violations approach 0 and objectives approach 140.



Final States: Plotted in Cell 14, showing final states clustered around ( (20, 20) ) within the feasible region (green area).

Notes





Environment: The custom LinearOptimizationEnv normalizes states to ([0, 1]^2) for DQN input but tracks unnormalized ([x, y] \in [0, 100]^2) in info for plotting.



Hyperparameters: Tuned for stability (e.g., LR=5e-4, EPSILON_DECAY=20000, PENALTY=100.0). Adjust in Cell 3 or 11 if needed.



Extending the Project: To test the trained policy or add visualizations (e.g., objective heatmap), refer to additional scripts or modify the notebook.

Author

Risshab Srinivas Ramesh

License

This project is licensed under the MIT License. See the LICENSE file for details.