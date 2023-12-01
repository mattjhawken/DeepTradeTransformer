from stock import Stock
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


def plot_params(a):
    model_name = "elysium-v1"
    with open(f"models/{model_name}.log") as f:
        a = f.read().split("\n")
        rewards = [float(c.split(",")[1]) for c in a[:-1]]

    # sample_input = torch.randn((32, 4, 186))
    # a.target_net(sample_input)
    # plt.imshow(activations, cmap="viridis", aspect="auto")
    plt.title(model_name)
    plt.show()


def load_stocks(tickers):
    for ticker in tickers:
        path = f"data/test/{ticker.lower()}"
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


agent = Agent("elysium-v1.0.1")
# agent.test()
agent.train()

# tickers = ["BA", "CNI", "GC=F", "GEO", "ILMN", "NG=F", "OHI", "REI-UN.TO", "SRU-UN.TO",
#            "TLT", "CM.TO", "SI=F", "XEG.TO", "XMA.TO"]
# tickers = ["XUT.TO", "ZUH.TO", "XRE.TO", "ZEB.TO", "CRSP", "MPW", "GF=F", "NTR.TO"]
# plot_stocks(tickers)
# plot_params(agent)
# load_stocks(tickers)
