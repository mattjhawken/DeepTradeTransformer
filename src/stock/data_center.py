import yfinance as yf


def load_stock(ticker, period, timeframe):
    # Load in data from yfinance API
    try:
        df = yf.download(tickers=ticker, interval=timeframe, period=period)
        return df

    except Exception as e:
        print(str(e))
