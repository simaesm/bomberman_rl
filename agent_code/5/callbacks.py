import os
import pickle
import random
import numpy as np
import torch
import heapq
from collections import deque
from .features import state_to_features
from .model import QNetwork

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Rest of your constants and utility functions...

def save_model(model, filename="my-saved-model.pt"):
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def setup(self):
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)
    self.bomb_buffer = 0

    # Initialize your QNetwork model here
    self.network = QNetwork(input_size=23, output_size=6)  # Adjust input_size and output_size

def act(self, game_state: dict) -> str:
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train:  # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <= eps:  # choose random action
            if eps > 0.1:
                if np.random.randint(10) == 0:  # old: 10 / 100 now: 3/4
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                    return action
                else:
                    action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                    self.logger.info(f"Waehle Aktion {action} komplett zufaellig")

                    return action

    action_prob = np.array(torch.softmax(Q, dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
    self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")

    return best_action

def get_reward(self, game_state: dict) -> float:
    # Extract relevant information from the game_state dictionary
    player_pos = np.array(game_state["self"][3])
    coins = np.array(game_state["coins"])
    crates = np.argwhere(np.array(game_state["field"]) == 1)
    destroyed_crates = self.destroyed_crates

    if len(coins) > 0:
        nearest_coin_distance = np.min(np.sum(np.abs(coins - player_pos), axis=1))
    else:
        nearest_coin_distance = 0

    if len(crates) > 0:
        nearest_crate_distance = np.min(np.sum(np.abs(crates - player_pos), axis=1))
    else:
        nearest_crate_distance = 0

    crate_difference = destroyed_crates - self.bomb_buffer

    reward = 0.5 * (1 / (nearest_coin_distance + 1)) + \
             0.3 * (1 / (nearest_crate_distance + 1)) + \
             0.2 * crate_difference

    return reward
