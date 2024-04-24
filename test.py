from src.rl.agent import Agent


agent = Agent("transforming-stonks-v1.0.1", tickers=["TSLA"])
agent.train()