# Learning-Based Path Planning for Optimal Terrain Traversal

This repository contains the official Python implementation for the paper: **"Learning-Based Path Planning for Optimal Terrain Traversal using Deep Reinforcement Learning"**.

The code provides a complete framework for training a Deep Q-Network (DQN) agent to navigate complex 2D grid environments. The agent learns to find paths that are not only short but also qualitatively superior, prioritizing smooth terrain and a comfortable ride over the absolute shortest route.

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/63ed431e-4b25-415c-82e3-1803109b5932" />


*Fig. 1: Qualitative comparison of paths on a 60x60 map. The DQN agent's path (blue) intelligently follows the high-smoothness "highway," while the traditional A* algorithm (red) takes a shorter but rougher path.*

---

## Key Features

- **Custom Grid Environment**: Procedurally generates maps with obstacles, speed breakers, and variable terrain smoothness.
- **Dual-Input DQN Agent**: The agent's neural network uses two inputs for decision-making: a local perception vector for immediate surroundings and a convolutional map for broader spatial awareness.
- **Advanced Training Techniques**: Implements Prioritized Experience Replay (PER) and pre-fills the replay buffer with expert demonstrations from a terrain-aware A* algorithm to accelerate and stabilize training.
- **Benchmarking Suite**: The agent's performance is comprehensively evaluated against standard pathfinding algorithms (A*, BFS, DFS) on a holistic set of metrics.

---

## Getting Started

### Prerequisites

You will need a Python environment (3.8 or newer) with the following libraries installed:
- `torch`
- `numpy`
- `matplotlib`
- `jupyter`

### Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file containing the libraries listed above.)*

---

## How to Run the Experiment

The entire experiment is self-contained within the `dqn-think-ai.ipynb` Jupyter Notebook.

1.  **Launch Jupyter**: Open a terminal in the project directory and run:
    ```bash
    jupyter notebook
    ```

2.  **Open the Notebook**: In the browser window that opens, click on `dqn-think-ai.ipynb`.

3.  **Run the Code**: To run the full simulation (training, evaluation, and plotting), simply click **"Cell" -> "Run All"** from the top menu.

The script will:
- Initialize the environment and all pathfinding algorithms.
- Run the DQN agent's training loop for the specified number of episodes.
- Evaluate the trained agent against the baselines on a new, unseen test map.
- Print a final summary table with all performance metrics.
- Save result visualizations to the local directory.

Key hyperparameters (e.g., `GRID_SIZE`, `NUM_EPISODES`) can be modified in the **"Hyperparameters and Setup"** section of the notebook.

---

## Citation

If you use this code or find our work helpful in your research, please consider citing our paper:

```bibtex
@inproceedings{alla2025learning,
  title={Learning-Based Path Planning for Optimal Terrain Traversal using Deep Reinforcement Learning},
  author={Alla, Ujwal Sai},
  year={2025}
}
