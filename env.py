import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os


class Env:
    def __init__(self, fee=0.001, trading_period=100):
        # General trading params
        self.fee = fee
        self.trading_period = trading_period
        self.stocks = []

        # Epoch specific trading params
        self.actions = []
        self.state_actions = [0 for _ in range(7)]
        self.rewards = []
        self.trades = []
        self.in_trade = False

        self.stock = None
        self.ind = 0
        self.start = 0
        self.end = 0

        self.load_data()
        self.reset()

    def load_stock(self, stock):
        self.stock = stock
        state = self.reset(flip=False, stock=False)
        return state

    def reset(self, flip=True, stock=True):
        if stock:
            self.stock = random.choice(self.stocks)

        if flip and random.random() > 0.5:
            self.stock.reverse()

        self.actions = []
        self.rewards = []
        self.trades = []
        self.in_trade = False

        self.start = random.sample(range(1, len(self.stock.closes) - self.trading_period - 1), 1)[0]
        self.end = self.start + self.trading_period
        self.ind = self.start

        # Feed in last 7 actions
        self.state_actions = [0 for _ in range(7)]
        self.stock.obs_space[self.ind, :, -1] = self.state_actions

        return self.stock.obs_space[self.ind, :, :]

    def step(self, action):
        """
        TODO: account for intra-day price swings around trading levels. Use smaller timeframe (1h) to see what was
            triggered first. Currently we are assuming the stop was hit first.
        :param action: a list of 3 trading targets (entry, target, stop loss) representing percentage
            difference from previous close (entry, target) and the entry (stop loss).
        :return: next state, actions, reward, done. For use by trading agent,
        """
        prev_close = self.stock.closes[self.ind - 1]
        p_change = np.log(self.stock.closes[self.ind] / prev_close)

        if not self.in_trade:
            if action == 1:
                p_change = np.log(self.stock.closes[self.ind] / self.stock.opens[self.ind])
                self.in_trade = True
                reward = p_change - self.fee
                self.trades.append((self.ind, 1))
            else:
                reward = -(p_change / 4)
        else:
            if action == 0:
                self.in_trade = False
                reward = -self.fee
                self.trades.append((self.ind, 0))
            else:
                reward = p_change

        self.ind += 1

        self.state_actions = [action for _ in range(7)]
        for i in range(1, 1 + min(7, len(self.actions))):
            self.state_actions[-i] = self.actions[-i]
        self.stock.obs_space[self.ind, :, -1] = self.state_actions

        self.rewards.append(reward)
        self.actions.append(action)

        done = False
        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, action, reward, done

    def get_cumulative_rewards(self):
        cumulative_reward = 1
        cumulative_rewards = []
        for i in range(len(self.rewards)):
            if self.actions[i] == 1 or (i > 0 and self.actions[i] != self.actions[i-1]):
                cumulative_reward *= (1 + self.rewards[i])
            cumulative_rewards.append(cumulative_reward)
        return cumulative_rewards, cumulative_reward

    def render(self, action_types=[], wait=False):
        da_range = list(range(self.start, self.end))

        fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

        ax1.scatter(da_range, self.stock.closes[da_range], c=self.actions, cmap="RdYlGn")
        ax1.set_title(f"{self.stock.ticker}: {self.start}-{self.ind}")

        for i in range(len(da_range[:-1])):
            other_ind = da_range[i]

            if action_types:
                if action_types[i] == 0:
                    ax1.axvspan(other_ind, other_ind+1, facecolor="whitesmoke", edgecolor="none", alpha=0.5)

        for i, trade in self.trades:
            ax2.axvline(i, c="r" if trade == 0 else "g")

        ax2.plot(da_range, self.rewards, linestyle='--', color='gray')

        plt.draw()
        if not wait:
            plt.pause(0.5)
        else:
            plt.waitforbuttonpress()
        plt.close()

    def load_data(self, p="data/train"):
        path = os.path.join(os.getcwd(), p)

        for file in os.listdir(path):
            new_path = path
            if ".DS" not in file:
                new_path = os.path.join(new_path, file)

                try:
                    file = open(new_path, "rb")
                    s = pickle.load(file)
                    # Adding vector for previous action
                    s.obs_space = np.concatenate([s.obs_space, np.zeros((s.obs_space.shape[0], s.obs_space.shape[1], 1))
                                                  ], axis=2)
                    # Temp fix for odd vector size
                    s.obs_space = s.obs_space[:, :, 1:]
                    self.stocks.append(s)
                except Exception as e:
                    print(str(e))

        print(f"Stocks loaded: {len(self.stocks)}.")
