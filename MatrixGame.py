import numpy as np
import random
from pettingzoo import ParallelEnv

class MatrixGame(ParallelEnv):
    def __init__(self, reward_matrix, agents, norm, judging=False, alpha=0.0, chi=0.01):
        self.agents = agents
        self.possible_agents = self.agents[:]
        # The reward matrix is the PD payoff matrix
        self.reward_matrix = reward_matrix
        self.norm = norm
        # To keep track of last opponents and actions of agents
        self.last_opponent = {}
        self.actions = {}
        # Introspective level
        self.alpha = alpha
        # Judging is True if it is a decentralized system
        self.judging = judging
        # Reputation assignment error
        self.chi = chi

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
        # Used for centralized system (pre-defined norm)
        if focal_action == 0 and opponent_state == 0:
            return self.norm[3]  # Bit 3
        elif focal_action == 0 and opponent_state == 1:
            return self.norm[2]  # Bit 2
        elif focal_action == 1 and opponent_state == 0:
            return self.norm[1]  # Bit 1
        else:  # (1,1)
            return self.norm[0]  # Bit 0

    def select_new_action(self, action_rule, focal_state, opponent_state):
        # Used for seeded agents that act following an action rule
        if focal_state == 0 and opponent_state == 0:
            return action_rule[3]  # Bit 3
        elif focal_state == 0 and opponent_state == 1:
            return action_rule[2]  # Bit 2
        elif focal_state == 1 and opponent_state == 0:
            return action_rule[1]  # Bit 1
        else:  # (1,1)
            return action_rule[0]  # Bit 0

    def step(self):
        # 1. Randomly select the pairings
        pairings = []
        players = self.agents.copy()
        num_agents = len(players)
        for _ in range(num_agents//2):
            index = random.randrange(len(players))
            elem1 = players.pop(index)

            index = random.randrange(len(players))
            elem2 = players.pop(index)

            pairings.append((elem1, elem2))

        # 2. Each pair plays the game and the update attributes are returned
        for pair in pairings:
            # Select an action based on the state of the opponent
            action1 = pair[0].select_action(self.states[pair[1]])
            action2 = pair[1].select_action(self.states[pair[0]])

            # Case of seeded agents
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

            # Introspective reward: the agent plays against himself
            intro_action1 = pair[0].select_action(self.states[pair[0]], False)
            intro_action2 = pair[1].select_action(self.states[pair[1]], False)

            # Case of seeded agents
            if intro_action1 == 5:
                intro_action1 = self.select_new_action(self.get_action_rules(5), self.states[pair[0]],
                                                       self.states[pair[0]])

            if intro_action2 == 5:
                intro_action2 = self.select_new_action(self.get_action_rules(5), self.states[pair[1]],
                                                       self.states[pair[1]])

            intro_reward1 = self.reward_matrix[intro_action1][intro_action1]
            intro_reward2 = self.reward_matrix[intro_action2][intro_action2]

            # Final reward with alpha being the level of introspection
            self.rewards[pair[0]] = (1 - self.alpha) * reward1 + self.alpha * intro_reward1
            self.rewards[pair[1]] = (1 - self.alpha) * reward2 + self.alpha * intro_reward2

            # For decentralized system: agents judge the behavior of others
            if self.judging:
                # Choose a judge that is not in the pair
                judge = random.choice([a for a in self.agents if a not in pair])

                # judge state corresponds to (focal action, opponent's reputation)
                # 0: (Defect,Bad), 1: (Defect,Good), 2: (Coop, Bad), 3: (Coop, Good)
                judge_state_1 = action1 * 2 + self.states[pair[1]] 
                judge_state_2 = action2 * 2 + self.states[pair[0]]

                state1 = judge.select_judge_action(judge_state_1)
                state2 = judge.select_judge_action(judge_state_2) 

                judge.store_judgement(judge_state_1, state1)
                judge.store_judgement(judge_state_2, state2)
            else:
                state1 = self.determine_state(action1, self.states[pair[1]])
                state2 = self.determine_state(action2, self.states[pair[0]])

            if (random.random() < self.chi):
                state1 = 1 - state1
            if (random.random() < self.chi):
                state2 = 1 - state2

            self.states[pair[0]] = state1
            self.states[pair[1]] = state2

        return self.states, self.rewards, self.actions

