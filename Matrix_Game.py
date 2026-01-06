import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from pettingzoo import ParallelEnv

num_agents = 10
num_rounds = 200
num_episodes = 500
b = 5           # Benefit
c = 1                    # Cost of cooperation

# Learning parameters
# chi = 10 / num_episodes    # Reputation assignment error
chi = 0.01


class MatrixGame(ParallelEnv):
    def __init__(self, reward_matrix, agents, norm, alpha=0.0):
        self.agents = agents
        self.possible_agents = self.agents[:]
        self.reward_matrix = reward_matrix
        self.norm = norm
        self.last_opponent = {}
        self.actions = {}
        self.alpha = alpha

    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.states = {agent: np.random.choice([0, 1]) for agent in self.agents}

        return self.states

    def get_action_rules(self, action_rule):
        # Convert action_rule_id to 4-bits
        bits = [(action_rule >> i) & 1 for i in range(4)]  # bits[0]=LSB, bits[3]=MSB
        return bits

    def determine_state(self, focal_action, opponent_state):
        if focal_action == 0 and opponent_state == 0:
            return self.norm[3]  # Bit 3
        elif focal_action == 0 and opponent_state == 1:
            return self.norm[2]  # Bit 2
        elif focal_action == 1 and opponent_state == 0:
            return self.norm[1]  # Bit 1
        else:  # (1,1)
            return self.norm[0]  # Bit 0

    def select_new_action(self, action_rule, focal_state, opponent_state):
        if focal_state == 0 and opponent_state == 0:
            return action_rule[3]  # Bit 3
        elif focal_state == 0 and opponent_state == 1:
            return action_rule[2]  # Bit 2
        elif focal_state == 1 and opponent_state == 0:
            return action_rule[1]  # Bit 1
        else:  # (1,1)
            return action_rule[0]  # Bit 0

    def step(self):
        pairings = []
        players = self.agents.copy()
        for _ in range(num_agents // 2):
            index = random.randrange(len(players))
            elem1 = players.pop(index)

            index = random.randrange(len(players))
            elem2 = players.pop(index)

            pairings.append((elem1, elem2))

        for pair in pairings:
            action1 = pair[0].select_action(self.states[pair[1]])
            action2 = pair[1].select_action(self.states[pair[0]])

            if action1 == 5:
                action1 = self.select_new_action(self.get_action_rules(5), self.states[pair[0]], self.states[pair[1]])

            if action2 == 5:
                action2 = self.select_new_action(self.get_action_rules(5), self.states[pair[1]], self.states[pair[0]])

            self.actions[pair[0]] = action1
            self.actions[pair[1]] = action2

            self.last_opponent[pair[0]] = pair[1]
            self.last_opponent[pair[1]] = pair[0]

            reward1 = self.reward_matrix[action1][action2]
            reward2 = self.reward_matrix[action2][action1]

            intro_action1 = pair[0].select_action(self.states[pair[0]])
            intro_action2 = pair[1].select_action(self.states[pair[1]])

            if intro_action1 == 5:
                intro_action1 = self.select_new_action(self.get_action_rules(5), self.states[pair[0]],
                                                       self.states[pair[0]])

            if intro_action2 == 5:
                intro_action2 = self.select_new_action(self.get_action_rules(5), self.states[pair[1]],
                                                       self.states[pair[1]])

            intro_reward1 = self.reward_matrix[intro_action1][intro_action1]
            intro_reward2 = self.reward_matrix[intro_action2][intro_action2]

            self.rewards[pair[0]] = (1 - self.alpha) * reward1 + self.alpha * intro_reward1
            self.rewards[pair[1]] = (1 - self.alpha) * reward2 + self.alpha * intro_reward2


            state1 = self.determine_state(action1, self.states[pair[1]])
            state2 = self.determine_state(action2, self.states[pair[0]])

            if (random.random() < chi):
                state1 = 1 - state1
            if (random.random() < chi):
                state2 = 1 - state2

            self.states[pair[0]] = state1
            self.states[pair[1]] = state2

        return self.states, self.rewards, self.actions

