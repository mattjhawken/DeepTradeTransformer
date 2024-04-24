from src.stock.data_center import load_stock
import numpy as np


class Stock:
    """
    Stock object that carries price and associated info, observation space for NN,
    along with methods for calculating various indicators
    """
    def __init__(self, ticker, period="6000d", timeframe="1d", window=364):
        self.ticker = ticker
        self.timeframe = timeframe
        self.period = period
        self.window = window

        # Loading in price data
        df = self.import_data()
        self.opens = df["Open"].to_numpy()
        self.highs = df["High"].to_numpy()
        self.lows = df["Low"].to_numpy()
        self.closes = df["Close"].to_numpy()
        self.volume = df["Volume"].to_numpy()
        self.dates = df.index

        # Observation space for neural network
        self.obs_space = None
        self.create_observation_space()

    def import_data(self, ticker=None, period=None, timeframe=None):
        """Import data from yfinance with Stock params if none are specified"""
        ticker = self.ticker if not ticker else ticker
        period = self.period if not period else period
        timeframe = self.timeframe if not timeframe else timeframe

        return load_stock(ticker, period, timeframe)

    def create_observation_space(self):
        """Create normalized price data for RL-environment. """
        daily = self.import_data(timeframe="1d")
        obs_space = []

        for i in range(self.window, len(self.closes)):
            # Create a state for each timestep past the defined date
            date = daily.index[i]
            daily_dates = daily.index[daily.index < date][-self.window:]
            daily_data = daily.loc[daily_dates]

            # # These values are 0-1 scaled representations of day of week, day of month, and day of year.
            # # Potentially useful information for the agent as humans look at daily, weekly, monthly charts
            # weeks = np.array([daily_dates[i].day_of_week/6 for i in range(len(daily_dates))])
            # months = np.array([daily_dates[i].day/daily_dates[i].daysinmonth for i in range(len(daily_dates))])
            # years = np.array([daily_dates[i].day_of_year/365.25 for i in range(len(daily_dates))])

            state = daily_data.iloc[-self.window:].to_numpy()[:, :4]
            state = (state - state.min()) / (state.max() - state.min())

            state = np.concatenate([
                state,
                # weeks[-self.window:, np.newaxis],
                # months[-self.window:, np.newaxis],
                # years[-self.window:, np.newaxis]
            ], axis=1)

            obs_space.append(state)

        self.opens = self.opens[self.window:]
        self.highs = self.highs[self.window:]
        self.lows = self.lows[self.window:]
        self.closes = self.closes[self.window:]
        self.volume = self.volume[self.window:]
        self.obs_space = np.array(obs_space).reshape((-1, 4, self.window))

    def reverse(self):
        """Method to reverse the entire stock data, potentially helps balance training"""
        self.opens = self.opens[::-1]
        self.highs = self.highs[::-1]
        self.lows = self.lows[::-1]
        self.closes = self.closes[::-1]
        self.volume = self.volume[::-1]
        self.obs_space = self.obs_space[::-1, :]

    def get_sma(self, data, mav=30):
        dat = []
        for i in range(len(data)):
            if i < mav:
                dat.append(np.mean(data[:i+1]))
            else:
                dat.append(np.mean(data[i-mav:i+1]))
        return dat

    def get_ema(self, data, period):
        ema = []
        for i in range(len(data)):
            if i == 0:
                ema.append(data[i])
            else:
                k = 2 / (period + 1)
                ema.append(data[i] * k + ema[-1] * (1 - k))
        return ema

    def get_sd(self, data, period):
        return np.array([np.std(data[i-period:i+1]) if i >= period else np.std(data[:i+1]) for i in range(len(data))])

    def get_cci(self, period):
        mean_prices = np.array([np.mean([self.highs[i], self.lows[i], self.closes[i]]) for i in range(len(self.lows))])
        mav_prices = self.get_sma(mean_prices, period)
        mean_dev = self.get_sma(np.abs(mean_prices - mav_prices), period)

        cci = []
        for i in range(len(mav_prices)):
            if i == 0:
                cci.append(0)
            elif mean_dev[i] == 0:
                cci.append(0)
            else:
                val = (mean_prices[i] - mav_prices[i]) / (0.015 * mean_dev[i])
                if val > 300:
                    val = 300
                elif val < -300:
                    val = -300
                cci.append(val)

        # Return normalized CCI
        return np.array((np.array(cci) - min(cci)) / (max(cci) - min(cci)))

    def get_rsi(self, period):
        # Calculate RSI
        rsi = [0 for _ in range(period)]
        for i in range(period, len(self.closes)):
            ups = []
            downs = []
            for n in range(i, i-period, -1):
                up = self.closes[n] - self.closes[n-1] if self.closes[n] > self.closes[n-1] else 0
                down = abs(self.closes[n] - self.closes[n-1]) if self.closes[n] < self.closes[n-1] else 0
                ups.append(up)
                downs.append(down)
            avg_up = np.mean(ups)
            avg_down = np.mean(downs)

            if avg_down == 0:
                rsi.append(100)
            else:
                rsi.append(100 - (100 / (1 + avg_up / avg_down)))

        # Return normalized RSI
        return np.array((np.array(rsi) - min(rsi[period+1:])) /
                        (max(rsi[period+1:]) - min(rsi[period+1:])))

    def get_bollinger_bands(self, data=None, term="long"):
        if data is None:
            data = self.closes

        if term == "xtralong":
            mav = self.get_sma(data, 200)
            sd = 3 * self.get_sd(data, 200)
        elif term == "long":
            mav = self.get_sma(data, 50)
            sd = 2.5 * self.get_sd(data, 50)
        elif term == "med":
            mav = self.get_sma(data, 20)
            sd = 2 * self.get_sd(data, 20)
        else:
            mav = self.get_sma(data, 10)
            sd = 1.5 * self.get_sd(data, 10)
        return mav + sd, mav - sd

    def dy(self, data, mav=None):
        if mav:
            data = self.moving_average(data, mav)
        dy = [0]
        for i in range(1, len(data)):
            dy.append((data[i] - data[i-1]) / data[i-1])
