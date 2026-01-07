import numpy as np
import random
from pettingzoo import ParallelEnv

class MatrixGame(ParallelEnv):
    def __init__(self, reward_matrix, agents, norm, judging=False, alpha=0.0, chi=0.01):


        np.random.seed(46)
        random.seed(46)

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
            intro_reward1 = 0
            intro_reward2 = 0
            if self.alpha != 0:
                intro_action1 = pair[0].select_action(self.states[pair[0]], True)
                intro_action2 = pair[1].select_action(self.states[pair[1]], True)

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





class Qlearner:
    def __init__(
            self,
            seeded=False,
            action_size=2,
            state_size=2,
            learning_rate=0.01,
            gamma=0.8,
            epsilon=0.1,
    ):
        self.action_size = action_size
        self.state_size = state_size

        self.seeded = seeded
        # initialize the Q-table: (State x Agent Action)
        self.qtable = np.zeros((self.state_size, self.action_size))

        self.learning_rate = learning_rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration

        # tracking rewards/progress:
        self.rewards_this_episode = []  # during an episode, save every time step's reward
        self.episode_total_rewards = []  # each episode, sum the rewards, possibly with a discount factor
        self.average_episode_total_rewards = []  # the average (discounted) episode reward to indicate progress

        self.state_history = []
        self.action_history = []

        # --- AJOUT (Section 5) ---
        # Q-table pour juger les autres agents (4 états possibles, 2 actions: rep 0 ou 1)
        self.judge_qtable = np.zeros((4, 2))
        self.judge_memory = []  # stocke les jugements sans reward immédiat
        # ------------------------

    def reset_agent(self):
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.judge_qtable = np.zeros((4, 2))  # --- AJOUT ---
        self.judge_memory = []                # --- AJOUT ---

    def select_greedy(self, state):
        # np.argmax(self.qtable[state]) will select first entry if two or more Q-values are equal, but we want true randomness:
        return np.random.choice(np.flatnonzero(np.isclose(self.qtable[state], self.qtable[state].max())))

    def select_action(self, state, exploration=True):
        if self.seeded:
            return 5
        if exploration and np.random.rand() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            action = self.select_greedy(state)
        self.state_history.append(state)
        self.action_history.append(action)
        return action

    # --- AJOUT (Section 5) ---
    def select_judge_action(self, judge_state):
        """Choisit la réputation (0 ou 1) à attribuer"""
        if np.random.rand() < self.epsilon:
            return random.randrange(2)
        return np.random.choice(np.flatnonzero(np.isclose(self.judge_qtable[judge_state], self.judge_qtable[judge_state].max())))

    def store_judgement(self, judge_state, judge_action):
        """Stocke le jugement (pas de reward immédiat)"""
        self.judge_memory.append((judge_state, judge_action))
    # ------------------------

    def update(self, state, action, new_state, reward, done):
        lr = self.learning_rate
        self.qtable[state, action] += lr * (
                    reward + (not done) * self.gamma * np.max(self.qtable[new_state]) - self.qtable[state, action])

        self.rewards_this_episode.append(reward)

        if done:
            # track total reward:
            episode_reward = self._calculate_episode_reward(self.rewards_this_episode, discount=False)
            self.episode_total_rewards.append(episode_reward)

            k = len(self.average_episode_total_rewards) + 1  # amount of episodes that have passed
            self._calculate_average_episode_reward(k, episode_reward)

            # --- AJOUT (Section 5) ---
            # Le reward final rétro-propage sur les jugements passés
            for s_j, a_j in self.judge_memory:
                self.judge_qtable[s_j, a_j] += lr * (
                    episode_reward - self.judge_qtable[s_j, a_j]
                )
            self.judge_memory = []
            # ------------------------

            # reset the rewards for the next episode:
            self.rewards_this_episode = []

    def _calculate_episode_reward(self, rewards_this_episode, discount=False):
        if discount:
            return sum([self.gamma ** i * reward for i, reward in enumerate(rewards_this_episode)])
        return sum(rewards_this_episode)

    def _calculate_average_episode_reward(self, k, episode_reward):
        if k > 1:  # running average is more efficient:
            average_episode_reward = (1 - 1 / k) * self.average_episode_total_rewards[-1] + episode_reward / k
        else:
            average_episode_reward = episode_reward
        self.average_episode_total_rewards.append(average_episode_reward)

    def print_rewards(self, episode, print_epsilon=True, print_q_table=True):
        # print("Episode ", episode + 1)
        print("Total (discounted) reward of this episode: ", self.episode_total_rewards[episode])
        print("Average total reward over all episodes until now: ", self.average_episode_total_rewards[-1])

        print("Epsilon:", self.epsilon) if print_epsilon else None
        print("Q-table: ", self.qtable) if print_q_table else None
