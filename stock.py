from data_center import load_stock
import yfinance as yf
import numpy as np


class Stock:
    def __init__(self, ticker, period, timeframe):
        self.ticker = ticker
        self.timeframe = timeframe
        self.period = period

        df = self.import_data()
        self.opens = df["Open"].to_numpy()
        self.highs = df["High"].to_numpy()
        self.lows = df["Low"].to_numpy()
        self.closes = df["Close"].to_numpy()
        self.volume = df["Volume"].to_numpy()
        self.dates = df.index

        self.obs_space = None
        self.create_observation_space()

    def import_data(self, ticker=None, period=None, timeframe=None):
        ticker = self.ticker if not ticker else ticker
        period = self.period if not period else period
        timeframe = self.timeframe if not timeframe else timeframe

        return load_stock(ticker, period, timeframe)

    def create_observation_space(self):
        # Add current day open to state (stock must make decision to trade at beginning of
        # trading day so you would have the data)
        _slice = 200
        daily = self.import_data(period="6000d", timeframe="1d")
        # weekly = self.import_data(period="6000d", timeframe="1wk")
        # monthly = self.import_data(period="6000d", timeframe="1mo")

        obs_space = []

        for i in range(_slice, len(self.closes)):
            # Create a state for each timestep past the defined date
            date = daily.index[i]

            daily_dates = daily.index[daily.index < date][-_slice:]
            # weekly_dates = weekly.index[weekly.index < date][-60:]
            # monthly_dates = monthly.index[monthly.index < date]

            # Remove last dates as they represent the current candle
            daily_data = daily.loc[daily_dates]
            # weekly_data = weekly.loc[weekly_dates][:-1]
            # monthly_data = monthly.loc[monthly_dates][:-1]

            weeks = np.array([daily_dates[i].day_of_week/6 for i in range(len(daily_dates))])
            months = np.array([daily_dates[i].day/daily_dates[i].daysinmonth for i in range(len(daily_dates))])
            years = np.array([daily_dates[i].day_of_year/365.25 for i in range(len(daily_dates))])

            state = daily_data.iloc[-_slice:].to_numpy()[:, :4]
            state = (state - state.min()) / (state.max() - state.min())

            state = np.concatenate([
                state,
                weeks[-_slice:, np.newaxis],
                months[-_slice:, np.newaxis],
                years[-_slice:, np.newaxis]
            ], axis=1)

            obs_space.append(state)

        self.opens = self.opens[_slice:]
        self.highs = self.highs[_slice:]
        self.lows = self.lows[_slice:]
        self.closes = self.closes[_slice:]
        self.volume = self.volume[_slice:]
        self.obs_space = np.array(obs_space).reshape((-1, 7, _slice))

    def reverse(self):
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
