import random
import os
from pathlib import Path
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
from train import DQN, ReplayBuffer, NeuralNetwork


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    file = Path("score.txt")
    if not file.is_file():
        seed_everything(seed=42)
        # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
        agent = DQN(number_of_layers = 8,
            number_of_neurons = [6, 128, 256, 256, 512, 256, 256, 128, 4],
            buffer_size = 60000, 
            batch_size = 1024, 
            gamma = 0.99, 
            epsilon_decay = 1200, 
            epsilon_decay_period = 18000,
            epsilon_min = 0.05, 
            epsilon_max = 1.0, 
            learning_rate = 0.001,
            loss = torch.nn.SmoothL1Loss(),
            gradient_steps = 5,
            update_target_freq = 800)
        agent.load()
        # Evaluate agent and write score.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
        with open(file="score.txt", mode="w") as f:
            f.write(f"{score_agent}\n{score_agent_dr}")
