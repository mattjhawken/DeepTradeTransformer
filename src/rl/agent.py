from src.ml.dqtn import DQTN
from src.rl.env import Env

from tqdm import tqdm
import numpy as np
import random
import pickle
import torch
import os


class Agent:
    def __init__(
        self,
        model_name="transforming-stonks",
        tickers=None,

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
    ):
        """
           Initializes the Agent with specific configuration for its trading model, learning parameters,
           and environment settings.
           """
        # Set up the trading environment
        self.env = Env(tickers, fee, trading_period)
        self.replay_mem = ReplayMemory(capacity)
        self.model_name = model_name

        # Learning and exploration parameters
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay = epsilon_decay
        self.capacity = capacity
        self.mini_batch_size = mini_batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.discount = discount

        # Initialize neural network models for policy and target
        self.target_net = DQTN(
            dims=self.env.stock.obs_space.shape,
            lr=lr,
            dropout=dropout,
            embeddings=embeddings,
            layers=layers,
            heads=heads,
            fwex=fwex,
            neurons=neurons,
            gamma=gamma
        )
        self.policy_net = self.target_net

        self.update_freq = update_freq
        self.n_eps = n_eps
        self.show_every = show_every
        self.render = render

        self.random_pos = 0
        self.random_n = 0

    def learn(self, step_count):
        """
        Performs a single step of training using a mini-batch from the replay memory.
        """
        states, actions, rewards, next_states = self.replay_mem.sample(self.mini_batch_size)
        next_states = torch.Tensor(next_states)
        rewards = torch.Tensor(rewards).unsqueeze(-1)

        current_qs = self.policy_net.forward(torch.Tensor(states))

        with torch.no_grad():
            future_qs = self.target_net.forward(next_states)
            target_qs = rewards + self.discount * future_qs

        self.policy_net.backward(current_qs, target_qs)

        if step_count % self.update_freq == 0:
            self.update_target()

        return self.policy_net.loss

    def train(self):
        """
        Conducts the training loop over a set number of episodes, managing exploration,
        updates, and logging.
        """
        model_info = f"====== Model Details: {self.model_name} ======\n" \
                     f"\033[1;37mModel Parameters:\033[0m\n" \
                     f"  \033[1;33mEmbeddings: {self.target_net.embeddings}\033[0m\n" \
                     f"  \033[1;33mLayers: {self.target_net.layers}\033[0m\n" \
                     f"  \033[1;33mHeads: {self.target_net.heads}\033[0m\n" \
                     f"  \033[1;33mFwex: {self.target_net.fwex}\033[0m\n" \
                     f"  \033[1;33mDropout: {self.target_net.dropout}\033[0m\n" \
                     f"  \033[1;33mNeurons: {self.target_net.neurons}\033[0m\n" \
                     f"  \033[1;33mLearning Rate: {self.lr}\033[0m\n" \
                     f"\n" \
                     f"\033[1;37mRL Parameters:\033[0m\n" \
                     f"  \033[1;36mUpdate Frequency: {self.update_freq}\033[0m\n" \
                     f"  \033[1;36mDecay: {self.decay}\033[0m\n" \
                     f"  \033[1;36mDiscount: {self.discount}\033[0m\n" \
                     f"  \033[1;36mCapacity: {self.capacity}\033[0m\n" \
                     f"\n" \
                     f"\033[1;37mTrading Parameters:\033[0m\n" \
                     f"  \033[1;35mFee: {self.env.fee}\033[0m\n" \
                     "====================================="

        print(model_info)
        self.fill_memory()

        path = f"models/{self.model_name}"

        with open(path + ".log", "w") as f:
            step_count = 0
            rewards = []
            cum_rewards = []
            losses = []

            try:
                for ep in range(self.n_eps):
                    done = False
                    state = self.env.reset()
                    loss = None
                    action_types = []
                    _rewards = []

                    while not done:
                        action, action_type = self.select_actions(state)
                        next_state, _, reward, done = self.env.step(action)
                        self.replay_mem.store(state, action, reward, next_state)
                        loss = self.learn(step_count)
                        action_types.append(action_type)
                        _rewards.append(reward)

                        state = next_state
                        step_count += 1

                    reward = np.round(np.sum(_rewards), 4)
                    rewards.append(reward)

                    cumulative_rewards, cumulative_reward = self.env.get_cumulative_rewards()

                    cum_rewards.append(cumulative_reward)
                    avg_reward = np.mean(rewards[-25:]).round(4)
                    avg_cum_reward = np.mean(cum_rewards[-25:]).round(4)
                    lr = self.target_net.optimizer.param_groups[0]['lr']
                    loss = loss.detach().numpy().round(4)
                    losses.append(loss)
                    avg_loss = np.mean(losses[-25:]).round(4)

                    f.write(f"{ep},{reward},{avg_reward},{cumulative_reward:.4f},{lr:.4e},{loss:.4f},{avg_loss:.4f},{self.epsilon}\n")
                    print(f"Ep: {ep}, Reward: {reward}, \033[93mReward (avg): {avg_reward}\033[0m, "
                          f"Performance: {cumulative_reward:.3f}, \033[92mPerformance (avg): {avg_cum_reward:.3f}\033[0m, "
                          f"Lr: {lr:.2e}, Loss: {loss:.4f}, \033[91mLoss (avg) {avg_loss:.3f}\033[0m, Epsilon: {self.epsilon}")

                    if ep % self.show_every == 0 and self.render:
                        self.env.render(action_types)

                    self.update_epsilon(ep)
                    self.policy_net.scheduler.step()

            except KeyboardInterrupt:
                print("Training interrupted, saving model...")
                self.save(self.model_name)

        self.save(self.model_name)

    def test(self):
        rewards = []
        p = "data/test"

        for file in os.listdir(p):
            new_path = p
            if ".DS" not in file:
                new_path = os.path.join(new_path, file)

                file = open(new_path, "rb")
                s = pickle.load(file)

                done = False
                state = self.env.load_stock(s)
                reward = 0

                while not done:
                    action, action_type = self.select_actions(state)
                    next_state, _, reward, done = self.env.step(action)
                    state = next_state

                rewards.append(reward)
                # self.env.render(True)
        return rewards

    def select_actions(self, state):
        """
        Selects actions based on the current state of the environment, using an epsilon-greedy strategy for exploration.
        """
        epsilon = max(self.epsilon, self.epsilon_min)

        if self.random_n != 0:
            self.random_n -= 1
            return self.random_pos - 1, 0
        elif random.random() < epsilon:
            # Return random trading values
            # return [random.uniform(0, 0.1), random.uniform(0, 0.2), random.uniform(0, 0.05)]
            self.random_n = random.sample([3, 5, 9], 1)[0]
            self.random_pos = random.sample([1, 2], 1)[0]
            return self.random_pos - 1, 0

        self.policy_net.eval()
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0)
            out = self.policy_net(state).squeeze()
            self.policy_net.train()
            return torch.argmax(out, dim=0), 1

    def fill_memory(self):
        """
        Pre-fills the replay memory with initial experiences by interacting with the environment using random actions.
        """
        for _ in tqdm(range(self.capacity // 3), desc="Initializing replay"):
            state = self.env.reset()
            done = False
            while not done:
                actions, _ = self.select_actions(state)
                next_state, action, reward, done = self.env.step(actions)
                self.replay_mem.store(state, actions, reward, next_state)
                state = next_state

    def update_epsilon(self, ep=0):
        """
         Updates the epsilon value for exploration based on the decay rate.
        """
        self.epsilon = max(self.epsilon_min, round(self.epsilon * self.decay, 4))

        if ep == 15:
            self.decay = 0.96

    def update_target(self):
        """
        Updates the target network with weights from the policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, model_name):
        """
        Saves the model state to disk.
        """
        torch.save(self.policy_net.state_dict(), f"models/{model_name}")

    def load(self, model_name):
        """
        Loads a model state from disk and sets the network to evaluation mode.
        """

        self.policy_net.load_state_dict(torch.load(f"models/{model_name}"))
        self.policy_net.eval()


class ReplayMemory:
    def __init__(self, capacity):
        """
        Initializes the ReplayMemory with a given capacity.
        """
        self.capacity = capacity
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.ind = 0

    def store(self, states, actions, rewards, next_states):
        """
        Stores or replaces an experience in the memory.
        """
        if len(self.states) < self.capacity:
            self.states.append(states)
            self.actions.append(actions)
            self.next_states.append(next_states)
            self.rewards.append(rewards)
        else:
            self.states[self.ind] = states
            self.actions[self.ind] = actions
            self.next_states[self.ind] = next_states
            self.rewards[self.ind] = rewards

        self.ind = (self.ind + 1) % self.capacity

    def sample(self, batchsize):
        """
        Randomly samples a batch of experiences from memory.
        """
        indices_to_sample = random.sample(range(len(self.states)), k=batchsize)

        states = np.array(self.states)[indices_to_sample]
        actions = np.array(self.actions)[indices_to_sample]
        next_states = np.array(self.next_states)[indices_to_sample]
        rewards = np.array(self.rewards)[indices_to_sample]

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.states)
