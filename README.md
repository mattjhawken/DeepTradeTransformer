## A Deep-RL Transformer-Based Trading Agent in PyTorch

This repository contains a trading agent that leverages deep-Q learning (RL) and an encoder-based transformer, built in PyTorch.

### Quickstart Guide

#### Requirements
- Python 3.9 or later

#### Clone the repository:
```bash
git clone https://github.com/mattjhawken/DeepTradeTransformer.git
cd DeepTradeTransformer
```

#### Set up a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Agent
To run the agent and start training, use the following command:

```bash
python main.py
```

### Key Input Parameters
The agent's behavior is controlled by several key parameters, detailed below:

- **tickers**: List of stock symbols to be used for trading simulations, e.g., ["AAPL", "GOOGL"].
- **model_name**: A unique identifier for saving/loading trained models.
- **embeddings**: The size of the embedding layer in the transformer model.
- **layers**: The number of transformer layers used in the model.
- **heads**: The number of attention heads in each transformer layer.
- **fwex**: Forward expansion size of the transformer's feed-forward network.
- **dropout**: Dropout rate used in the transformer model to prevent overfitting.
- **neurons**: The number of neurons in the fully connected layers of the network.
- **lr (Learning Rate)**: Controls how much to change the model in response to the estimated error each time the model weights are updated.
- **gamma**: The discount factor used in the reinforcement learning update rule.
- **mini_batch_size**: Size of batches taken from the replay memory for training.
- **epsilon_max**: Initial value of ε for the ε-greedy policy, controlling exploration.
- **epsilon_min**: Minimum value of ε after decay, determining the amount of exploration.
- **epsilon_decay**: The factor by which ε is decreased during training.
- **discount**: Discount factor for future rewards in the Q-learning update.
- **capacity**: The capacity of the replay memory.
- **n_eps**: Number of episodes to train over.
- **update_freq**: Frequency (in steps) at which the target network is updated.
- **show_every**: Frequency (in episodes) at which training episodes are rendered/visualized.
- **render**: Boolean flag to turn on/off rendering of the trading environment.
- **fee**: Trading fee percentage used in simulations.
- **trading_period**: Number of time steps each trading episode lasts.

### Disclaimer:
This project was not intended for public use, as a result the degree of commenting and organization is likely horrible (non-existent). I will try and go through it ASAP to clean it up and make some improvements.
```