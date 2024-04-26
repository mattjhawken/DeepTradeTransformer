import matplotlib.pyplot as plt
from src.stock.stock import Stock
import numpy as np
import random
import pickle
import os


class Env:
    def __init__(self, tickers=None, fee=0.001, trading_period=100):
        # Initializes the trading environment with optional tickers, a trading fee, and a specified trading period.
        self.tickers = tickers
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

        self.load_data(tickers=tickers)
        self.reset()

    def load_stock(self, stock):
        # Sets the current stock to the provided stock and resets the environment state.
        self.stock = stock
        state = self.reset(flip=False, stock=False)
        return state

    def reset(self, flip=True, stock=True):
        # Resets the environment to a new trading session. Optionally flips the stock data and selects a random stock.
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

        # # Feed in last 7 actions
        # self.state_actions = [0 for _ in range(7)]
        # self.stock.obs_space[self.ind, :, -1] = self.state_actions

        return self.stock.obs_space[self.ind, :, :]

    def step(self, action):
        """
        Advances the environment by one step based on the action taken. Calculates the change in price as reward and manages trades.
        :param action: The trading action to take (e.g., buy, hold, sell).
        :return: next state, action, reward, and a boolean indicating if the session is done.
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

        # self.state_actions = [action for _ in range(7)]
        # for i in range(1, 1 + min(7, len(self.actions))):
        #     self.state_actions[-i] = self.actions[-i]
        # self.stock.obs_space[self.ind, :, -1] = self.state_actions

        self.rewards.append(reward)
        self.actions.append(action)

        done = False
        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, action, reward, done

    def get_cumulative_rewards(self):
        # Computes the cumulative rewards over the trading period. (current not used for training)
        cumulative_reward = 1
        cumulative_rewards = []
        for i in range(len(self.rewards)):
            if self.actions[i] == 1 or (i > 0 and self.actions[i] != self.actions[i-1]):
                cumulative_reward *= (1 + self.rewards[i])
            cumulative_rewards.append(cumulative_reward)

        return cumulative_rewards, cumulative_reward

    def render(self, action_types=[], wait=False):
        # Renders the stock prices and trades over the trading period.
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

    def load_data(self, p="data/train", tickers=None):
        if tickers is None:
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
        else:
            for t in tickers:
                s = Stock(t)
                self.stocks.append(s)

        print(f"Stocks loaded: {len(self.stocks)}.")
