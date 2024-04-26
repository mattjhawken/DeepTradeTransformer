from src.rl.agent import Agent


if __name__ == "__main__":
    agent = Agent(
        model_name="transforming-stonks-v1.0.1",
        tickers=["AAPL", "GOOGL"],  # Example tickers
        embeddings=256,
        layers=1,
        heads=4,
        fwex=128,
        dropout=0.1,
        neurons=1024,
        lr=1e-4,
        gamma=0.9,
        mini_batch_size=16,
        epsilon_max=1,
        epsilon_min=0.005,
        epsilon_decay=0.83,
        discount=0.98,
        capacity=10_000,
        n_eps=1_000,
        update_freq=500,
        show_every=5,
        render=True,
        fee=0.005,
        trading_period=300,
    )
    agent.train()
