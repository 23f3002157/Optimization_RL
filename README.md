# Optimization with Reinforcement Learning

This repository explores how **Reinforcement Learning (RL)**, specifically **Deep Q-Networks (DQN)**, can be applied to solve **constrained optimization problems**.  
It includes both **two-variable** and **three-variable** optimization problem variants, implemented in Python with custom environments, neural networks, and training workflows. For more details, refer Optimization_RL.pdf

---

## 📖 Introduction

Optimization is at the core of challenges across science and industry, from routing logistics to designing resilient systems.  
Traditional approaches (linear programming, heuristics, genetic algorithms) often struggle with **complex or dynamic problems**.  

**Reinforcement Learning (RL)** provides a flexible alternative:  
- An **agent** interacts with an **environment**.  
- It learns through **trial and error** guided by **rewards and penalties**.  
- Over time, it develops a **policy** that maps states to optimal actions.  

---

## 🧠 Deep Q-Networks (DQN)

DQN combines **Q-learning** with deep neural networks to approximate Q-values:  
- **Input:** state vector (e.g., \(x, y\) or \(x, y, z\)).  
- **Output:** estimated Q-values for all possible actions.  
- The agent selects actions using an **ϵ-greedy policy**, balancing exploration and exploitation.  

This architecture was famously used by **DeepMind** to achieve human-level control of Atari games.  

---

### Using this Repository

#### 1. Clone the Repository:

```
git clone https://github.com/23f3002157/Optimization_RL.git
```

#### 2. Navigate to the corresponding Directory

To run the two dimenional optimisation problem: 
```
cd Two_Variables
```

To run the three dimenional optimisation problem: 
```
cd Three_Variables
```
#### 3. Follow the README.md file for instruction under the corresponding directory.

---
### Authors
ForDr. Prasant Misra and Risshab Srinivas Ramesh