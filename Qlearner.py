import numpy as np
import random

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