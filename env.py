import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import os


class Env1:
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
        self.len = 0
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

        self.len = len(self.stock.closes)
        self.start = random.sample(range(1, len(self.stock.closes) - self.trading_period - 1), 1)[0]
        self.end = self.start + self.trading_period
        self.ind = self.start

        # Feed in last 7 actions
        # self.state_actions = [0 for _ in range(7)]
        # self.stock.obs_space[self.ind, :, -1] = self.state_actions

        return self.stock.obs_space[self.ind, :, :]

    def step(self, actions):
        """
        TODO: account for intra-day price swings around trading levels. Use smaller timeframe (1h) to see what was
            triggered first. Currently we are assuming the stop was hit first.
        :param actions: a list of 3 trading targets (entry, target, stop loss) representing percentage
            difference from previous close (entry, target) and the entry (stop loss).
        :return: next state, actions, reward, done. For use by trading agent,
        """
        prev_close = self.stock.closes[self.ind - 1]
        entry, target, stop = actions

        entry = prev_close - (entry * prev_close)
        target = entry + (target * prev_close)
        stop = entry - (stop * prev_close)

        prediction_ind = self.ind

        reward = 0
        done = False

        # Traverse timesteps until trade is executed and finished, or we reach the end of the period
        while True:
            if self.ind > self.end - 1:
                _exit = self.ind - 1
                break

            if self.in_trade:
                # If hit target
                if target <= self.stock.highs[self.ind]:
                    reward = (target - entry) / entry - self.fee
                    self.in_trade = False

                    _exit = self.ind
                    self.rewards.append(reward)
                    break

                # If hit stop
                elif stop >= self.stock.lows[self.ind]:
                    reward = (stop - entry) / entry - self.fee
                    self.in_trade = False

                    _exit = self.ind
                    self.rewards.append(reward)
                    break
                else:
                    reward = -0.0001
                    self.rewards.append(reward)
            else:
                # Order triggered
                if entry >= self.stock.lows[self.ind]:

                    # In the case that we drop below the stop loss in the same period
                    if stop >= self.stock.lows[self.ind]:
                        reward = (stop - entry) / entry - self.fee

                        _exit = self.ind
                        self.rewards.append(reward)
                        break
                    else:
                        reward = - self.fee
                        self.in_trade = True
                        self.rewards.append(reward)

                # Still waiting, light penalty applied
                else:
                    reward = -0.0001
                    self.rewards.append(reward)

                # # If price is above current, scrap trade
                # elif target <= self.stock.highs[self.ind]:
                #     reward -= self.fee
                #     _exit = self.ind
                #     self.rewards.append(reward)
                #     break

            self.ind += 1

        self.actions.append((prediction_ind, _exit, entry, target, stop))

        # Update environment params
        self.ind += 1

        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, actions, reward, done

    def render(self, wait=False):
        da_range = list(range(self.start, self.end))

        fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        for i in da_range:
            oc = (self.stock.closes[i], self.stock.opens[i])
            hl = (self.stock.highs[i], self.stock.lows[i])
            ax1.plot((i, i), oc, c="g" if oc[1] > oc[0] else "r", linewidth=1.5)
            ax1.plot((i, i), hl, linewidth=0.3)

        entries = []
        targets = []
        stops = []
        for start, end, e, t, s in self.actions:
            for n in range(start, end+1):
                entries.append(e)
                targets.append(t)
                stops.append(s)

        ax1.plot(da_range, entries, c="b")
        ax1.plot(da_range, targets, c="g")
        ax1.plot(da_range, stops, c="r")

        ax1.set_title(f"{self.stock.ticker}: {self.stock.dates[self.start]}-{self.stock.dates[self.ind]}")

        ax2.plot(da_range, np.cumsum(self.rewards), linestyle='--', c='gray')

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


class Env(Env1):
    def __init__(self, fee=0.001, trading_period=100):
        super().__init__(fee, trading_period)

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
        self.len = 0
        self.ind = 0
        self.start = 0
        self.end = 0

        self.load_data()
        self.reset()

    def reset(self, flip=True, stock=True):
        if stock:
            self.stock = random.choice(self.stocks)

        if flip and random.random() > 0.5:
            self.stock.reverse()

        self.actions = []
        self.rewards = []
        self.trades = []
        self.in_trade = False

        self.len = len(self.stock.closes)
        self.start = random.sample(range(1, len(self.stock.closes) - self.trading_period - 1), 1)[0]
        self.end = self.start + self.trading_period
        self.ind = self.start

        return self.stock.obs_space[self.ind, :, :]

    def step(self, actions):
        """
        TODO: account for intra-day price swings around trading levels. Use smaller timeframe (1h) to see what was
            triggered first. Currently we are assuming the stop was hit first.
        :param actions: a list of 3 trading targets (entry, target, stop loss) representing percentage
            difference from previous close (entry, target) and the entry (stop loss).
        :return: next state, actions, reward, done. For use by trading agent,
        """
        prev_close = self.stock.closes[self.ind - 1]
        entry, target, stop = actions

        entry = prev_close - (entry * prev_close)
        target = entry + (target * prev_close)
        stop = entry - (stop * prev_close)

        done = False

        if self.in_trade:
            # If hit target
            if target <= self.stock.highs[self.ind]:
                reward = (target - prev_close) / prev_close - self.fee
                self.in_trade = False
                self.rewards.append(reward)

            # If hit stop
            elif stop >= self.stock.lows[self.ind]:
                reward = (stop - prev_close) / prev_close - self.fee
                self.in_trade = False
                self.rewards.append(reward)

            else:
                reward = (self.stock.closes[self.ind] - prev_close) / prev_close
                self.rewards.append(reward)
        else:
            # Order triggered
            if entry >= self.stock.lows[self.ind]:
                # In the case that we drop below the stop loss in the same period
                if stop >= self.stock.lows[self.ind]:
                    reward = (stop - entry) / entry - self.fee
                    self.rewards.append(reward)
                else:
                    reward = (self.stock.closes[self.ind] - entry) / entry - self.fee
                    self.in_trade = True
                    self.rewards.append(reward)

            # Still waiting, light penalty applied
            else:
                reward = -0.0001
                self.rewards.append(reward)

        self.actions.append((entry, target, stop))

        # Update environment params
        self.ind += 1

        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, actions, reward, done

    def render(self, wait=False):
        da_range = list(range(self.start, self.end))

        fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        for i in da_range:
            oc = (self.stock.closes[i], self.stock.opens[i])
            hl = (self.stock.highs[i], self.stock.lows[i])
            ax1.plot((i, i), oc, c="g" if oc[1] > oc[0] else "r", linewidth=1.5)
            ax1.plot((i, i), hl, linewidth=0.3)

        entries = []
        targets = []
        stops = []
        for e, t, s in self.actions:
            entries.append(e)
            targets.append(t)
            stops.append(s)

        ax1.plot(da_range, entries, c="b")
        ax1.plot(da_range, targets, c="g")
        ax1.plot(da_range, stops, c="r")

        ax1.set_title(f"{self.stock.ticker}: {self.stock.dates[self.start]}-{self.stock.dates[self.ind]}")

        ax2.plot(da_range, np.cumsum(self.rewards), linestyle='--', c='gray')

        plt.draw()
        if not wait:
            plt.pause(0.5)
        else:
            plt.waitforbuttonpress()

        plt.close()
