from src.rl.agent import Agent


if __name__ == "__main__":
    model_name = "transforming-stonks-v1.0.1"

    # Trading parameters
    tickers = ["TSLA", "CNI", "COST", "TLT", "TD"]
    fee = 0.005
    trading_period = 250

    # Model parameters
    embeddings = 256
    layers = 1
    heads = 4
    fwex = 128
    dropout = 0.1
    neurons = 1024
    lr = 1e-4
    gamma = 0.9

    # RL parameters
    mini_batch_size = 16
    epsilon_max = 1
    epsilon_min = 0.005
    epsilon_decay = 0.83
    discount = 0.98
    capacity = 10_000
    n_eps = 1_000
    update_freq = 100
    show_every = 5
    render = True

    agent = Agent(
        model_name=model_name,
        tickers=tickers,
        fee=fee,
        trading_period=trading_period,
        embeddings=embeddings,
        layers=layers,
        heads=heads,
        fwex=fwex,
        dropout=dropout,
        neurons=neurons,
        lr=lr,
        gamma=gamma,
        mini_batch_size=mini_batch_size,
        epsilon_max=epsilon_max,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        discount=discount,
        capacity=capacity,
        n_eps=n_eps,
        update_freq=update_freq,
        show_every=show_every,
        render=render,
    )

    agent.train()
