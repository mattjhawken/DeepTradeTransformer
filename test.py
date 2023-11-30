from stock import Stock
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch


def plot_params(a):
    activations = []

    def hook_function(module, _input, _output):
        activations.append(_output.detach().cpu().numpy())

    hook_handles = []
    for layer in a.target_net.children():
        handle = layer.register_forward_hook(hook_function)
        hook_handles.append(handle)

    sample_input = torch.randn((32, 4, 186))
    a.target_net(sample_input)

    activations = np.array([list(activations[i]) for i in range(len(activations))])
    plt.imshow(activations, cmap="viridis", aspect="auto")
    plt.show()


def load_stocks(tickers):
    for ticker in tickers:
        path = f"data/assets/{ticker.lower()}"
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


agent = Agent("elysium-v1")
agent.train()

tickers = ["BA", "CNI", "GC=F", "GEO", "ILMN", "NG=F", "OHI", "REI-UN.TO", "SRU-UN.TO",
           "TLT", "MPW", "CRSP"]
# plot_stocks(tickers)
# plot_params(agent)
# load_stocks(tickers)
