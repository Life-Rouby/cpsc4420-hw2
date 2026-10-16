# CPSC 4420 Assignment 2: Q-Learning on Taxi

Tabular Q-learning agent for the Gymnasium Taxi-v4 environment, with experiments on the learning rate, discount factor, and epsilon decay.

## Setup

Requires Python 3.10 or newer.

```
git clone https://github.com/Life-Rouby/cpsc4420-hw2.git
cd cpsc4420-hw2
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

```
python q_learning.py
```

This trains the baseline agent and runs all experiments. It takes about 1-2 minutes. Results print to the terminal and plots are saved to the results folder. Hyperparameters can be changed at the top of q_learning.py.

## Files

- q_learning.py: Q-learning agent, experiments, and evaluation
- requirements.txt: required Python packages
- results/: plots and the saved Q-table of the best agent
- videos/: video of the trained agent

## Results

The baseline agent (alpha = 0.1, gamma = 0.99, epsilon decay = 0.995) learns to deliver every passenger in about 13 steps after roughly 1,500 episodes. The best configuration (alpha = 0.5, gamma = 0.99, epsilon decay = 0.99) reaches the same final performance about 2.8 times faster. Full results and discussion are in the PDF report.

## Video

videos/taxi_agent.mp4