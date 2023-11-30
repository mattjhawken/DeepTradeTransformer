import yfinance as yf


def load_stock(ticker, period, timeframe):
    try:
        df = yf.download(tickers=ticker, interval=timeframe, period=period)
        return df

    except Exception as e:
        print(str(e))


# def create_observation_space(ticker, period):
#     daily = load_stock(ticker, period, "1d")
#     weekly = load_stock(ticker, period, "1wk")
#     obs_space = []
#
#     for i in range(1000, len(daily))
#
