from stock import Stock
import matplotlib.pyplot as plt
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
        self.rewards = []
        self.trades = []
        self.in_trade = False
        self.profit = 1

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
        self.profit = 1

        self.start = random.sample(range(1, len(self.stock.closes) - self.trading_period - 1), 1)[0]
        self.end = self.start + self.trading_period
        self.ind = self.start

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
        p_change = (self.stock.closes[self.ind] - prev_close) / prev_close

        if not self.in_trade:
            if action == 1:
                p_change = (self.stock.closes[self.ind] - self.stock.opens[self.ind]) / self.stock.opens[self.ind]
                self.in_trade = True
                self.profit *= (1 - self.fee + p_change)
            else:
                self.profit *= (1 - (p_change/5))
        else:
            if action == 0:
                self.in_trade = False
                self.profit *= (1 - self.fee)
            else:
                self.profit *= (1 + p_change)

        self.rewards.append(self.profit)
        self.actions.append(action)
        self.ind += 1

        done = False
        if self.ind >= self.end or self.ind == len(self.stock.closes) - 1:
            done = True

        next_state = self.stock.obs_space[self.ind, :, :]

        return next_state, action, self.profit, done

    def render(self, wait=False):
        fig, [ax1, ax2] = plt.subplots(2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        da_range = list(range(self.start, self.end))

        ax1.set_title(f"{self.stock.ticker}: {self.start}-{self.ind}")

        ax1.scatter(da_range, self.stock.closes[da_range], c=self.actions, cmap="summer")

        ax2.set_title("P/L")
        ax2.plot(da_range, self.rewards)

        plt.draw()
        if not wait:
            plt.pause(0.5)
        else:
            plt.waitforbuttonpress()
        plt.close()

    def load_data(self, p="data/assets"):
        path = os.path.join(os.getcwd(), p)

        for file in os.listdir(path):
            new_path = path
            if ".DS" not in file:
                new_path = os.path.join(new_path, file)

                try:
                    file = open(new_path, "rb")
                    s = pickle.load(file)
                    self.stocks.append(s)
                except Exception as e:
                    print(str(e))

        print(f"Stocks loaded: {len(self.stocks)}.")

    def get_data(self, tickers):
        path = os.path.join(os.getcwd(), "data/stonks")

        for ticker in tickers:
            if ticker.lower() not in path:
                s = Stock(ticker, "6000d")
                file = open(os.path.join(path, ticker.lower()), "wb")
                pickle.dump(s, file)
