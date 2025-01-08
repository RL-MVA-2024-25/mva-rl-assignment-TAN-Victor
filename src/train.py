from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV

import torch
import random
import os
import numpy as np

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class NeuralNetwork(torch.nn.Module):
    def __init__(self, number_of_layers: int, number_of_neurons: list, leaky_relu: float = 0):
        super().__init__()
        if number_of_layers != len(number_of_neurons) - 1:
            raise ValueError(
                f"The length of the list of neurons should be equal to the number of layers minus one. Got {number_of_layers} layers and {len(number_of_neurons)} neurons."
            )
        self.layers = torch.nn.ModuleList()
        self.leaky_relu = leaky_relu
        for i in range(number_of_layers - 1):
            self.layers.append(torch.nn.Linear(number_of_neurons[i], number_of_neurons[i + 1]))
            self.layers.append(torch.nn.LeakyReLU(self.leaky_relu))
        self.layers.append(torch.nn.Linear(number_of_neurons[-2], number_of_neurons[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

        

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.device = DEVICE

    def append(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.index] = (
            torch.tensor(state, dtype=torch.float32).to(self.device),
            torch.tensor(action, dtype=torch.long).to(self.device),
            torch.tensor(reward, dtype=torch.float32).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.float32).to(self.device),
        )
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return list(map(torch.stack, zip(*batch)))
    
    def __len__(self):
        return len(self.buffer)
    


class ProjectAgent():
    def __init__(self):
        self.device = DEVICE
        self.number_of_layers = 8
        self.number_of_neurons = [6, 128, 256, 256, 512, 256, 256, 128, 4]
        self.model = NeuralNetwork(self.number_of_layers, self.number_of_neurons).to(self.device)
        self.target_model = NeuralNetwork(self.number_of_layers, self.number_of_neurons).to(self.device)
        self.target_model.eval()
        self.update_target_freq = 500
        self.update_tau = 0.0008
        self.path = "models"

        self.buffer_size = 20000
        self.buffer = ReplayBuffer(self.buffer_size)
        self.batch_size = 512
        self.gamma = 0.98
        self.epsilon_min = 0.05
        self.epsilon_max = 1.0
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / 20000
        self.epsilon_decay = 600

        self.gradient_steps = 5
        self.learning_rate = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20 * 200 * self.gradient_steps, gamma = 0.97)
        self.loss = torch.nn.SmoothL1Loss()

    def act(self, observation, use_random=False):
        # if use_random and random.random() < self.epsilon:     # Never used 
        #     return random.randint(0, 1)
        with torch.no_grad():
            return torch.argmax(self.model(torch.tensor(observation).float().to(self.device))).item()

    def save(self, path):
        string_name = path + f"/{self.number_of_layers}_{self.buffer_size}_{self.batch_size}_{self.gamma}_{self.epsilon_step}_{self.epsilon_decay}_{self.epsilon_min}_{self.learning_rate}.pt"
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), string_name)

    def load(self, path=None):
        if path:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            string_name = self.path + f"/{self.number_of_layers}_{self.buffer_size}_{self.batch_size}_{self.gamma}_{self.epsilon_step}_{self.epsilon_decay}_{self.epsilon_min}_{self.learning_rate}.pt"
            if not os.path.exists(string_name):
                print(f"No model found at {string_name}.")
            else:
                print("Model found, loading")
                self.model.load_state_dict(torch.load(string_name, map_location=self.device, weights_only=True))

    def gradient_step(self):
        if len(self.buffer) > self.batch_size:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            Q_max_next = self.target_model(next_states).max(1)[0].detach()
            update = torch.addcmul(rewards, 1 - dones, Q_max_next, value = self.gamma)
            Q = self.model(states).gather(1, actions.long().unsqueeze(1))
            loss = self.loss(Q, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def train(self, nb_episode: int):
        episode_return = []
        epsilon = self.epsilon_max
        step = 0
        best_return = -float("inf")
        eval_return = []
        episode = 0

        try:
            state, _ = env.reset()
            episode_cum_reward = 0
            while episode < nb_episode:
                if step > self.epsilon_decay:
                    epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
    
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state, use_random=False)
    
                next_state, reward, done, trunc, _ = env.step(action)
                self.buffer.append(state, action, reward, next_state, done)
                episode_cum_reward += reward
    
                for _ in range(self.gradient_steps):
                    self.gradient_step()
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
    
                step += 1
                if done or trunc:
                    evaluation_return = evaluate_HIV(self, nb_episode=1)
                    if evaluation_return > best_return:
                        best_return = evaluation_return
                        self.save("models")
    
                    # pbar.set_postfix({
                    #     "Epsilon": f"{epsilon:2.2f}",
                    #     "Buffer Size": f"{len(self.buffer):5d}",
                    #     "Cum-Return": f"{episode_cum_reward:4.2f}",
                    #     "Eval Score": f"{evaluation_return:4.2f}",
                    #     "Learning Rate": f"{self.scheduler.get_last_lr()[0]:.6f}"
                    # })
                    episode += 1
                    
                    episode_return.append(episode_cum_reward)
                    eval_return.append(evaluation_return)
                    state, _ = env.reset()
                    episode_cum_reward = 0
    
                else:
                    state = next_state
        except KeyboardInterrupt:
            print("Training interrupted. Saving current state")
            self.save("models_interrupted")
            return episode_return, eval_return

        return episode_return, eval_return