from src.stock.stock import Stock
import matplotlib.pyplot as plt
import pickle


def plot_params(model_name):
    with open(f"models/{model_name}.log") as f:
        a = f.read().split("\n")
        rewards = [float(c.split(",")[1]) for c in a[:-1]]
        loss = [float(c.split(",")[3]) for c in a[:-1]]
        lr = [float(c.split(",")[2]) for c in a[:-1]]

    fig, [ax1, ax2, ax3] = plt.subplots(3, sharex=True)
    plt.title(model_name)
    ax1.plot(loss)
    ax1.set_title("Loss")
    ax2.plot(lr)
    ax2.set_title("LR")
    ax3.plot(rewards)
    ax3.set_title("Reward")
    plt.show()


def load_stocks(tickers):
    for ticker in tickers:
        path = f"data/train/{ticker.lower()}"
        s = Stock(ticker, "6000d", "1d")
        file = open(path, "wb")
        pickle.dump(s, file)


def plot_stocks(tickers):
    for ticker in tickers:
        s = Stock(ticker, "6000d", "1d")
        fig = plt.figure(figsize=(12, 8))
        plt.title(s.ticker)

        for i in range(len(s.closes)):
            plt.plot((i, i), (s.opens[i], s.closes[i]), linewidth=2)
            plt.plot((i, i), (s.highs[i], s.lows[i]), linewidth=0.4)

        plt.show()
