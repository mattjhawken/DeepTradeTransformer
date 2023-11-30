from dqtn import DQTN
from env import Env

from tqdm import tqdm
import numpy as np
import random
import torch


class Agent:
    def __init__(
            self,
            model_name="elysium",

            embeddings=128,
            layers=1,
            heads=4,
            fwex=256,
            dropout=0.1,
            neurons=512,
            lr=1e-4,
            mini_batch_size=32,

            epsilon_max=1,
            epsilon_min=0.005,
            epsilon_decay=0.87,
            discount=0.98,
            capacity=10_000,

            n_eps=100,
            update_freq=500,
            show_every=5,
            render=True,

            fee=0.002,
            trading_period=150,
    ):
        self.env = Env(fee, trading_period)
        self.replay_mem = ReplayMemory(capacity)
        self.model_name = model_name

        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay = epsilon_decay
        self.capacity = capacity
        self.mini_batch_size = mini_batch_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = lr
        self.discount = discount

        self.target_net = DQTN(
            dims=self.env.stock.obs_space.shape,
            lr=lr,
            dropout=dropout,
            embeddings=embeddings,
            layers=layers,
            heads=heads,
            fwex=fwex,
            neurons=neurons
        )
        self.policy_net = self.target_net

        self.update_freq = update_freq
        self.n_eps = n_eps
        self.show_every = show_every
        self.render = render

        self.random_pos = 0
        self.random_n = 0

    def learn(self, step_count):
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
        print(f"====== Model Details: {self.model_name} ======")
        print(f"Decay: {self.decay}")
        print(f"Learning Rate: {self.lr}")
        print(f"Discount: {self.discount}")
        print(f"Capacity: {self.capacity}")
        print("=====================================")
        self.fill_memory()
        path = f"data/{self.model_name}"

        with open(path + ".log", "a") as f:
            step_count = 0
            rewards = []
            data = []
            losses = []

            for ep in range(self.n_eps):
                done = False
                state = self.env.reset()
                pl = 1
                loss = None

                while not done:
                    action = self.select_actions(state)
                    next_state, _, reward, done = self.env.step(action)
                    self.replay_mem.store(state, action, reward, next_state)
                    loss = self.learn(step_count)

                    pl = reward
                    state = next_state
                    step_count += 1

                # Save train data
                reward = round(pl, 4)
                data.append([ep, loss, reward])
                rewards.append(reward)
                avg_reward = np.mean(rewards[-25:]).round(4)
                lr = self.target_net.optimizer.param_groups[0]['lr']
                loss = loss.detach().numpy().round(4)
                losses.append(loss)
                avg_loss = np.mean(losses[-25:]).round(4)
                self.target_net.scheduler.step(ep)
                self.update_epsilon(ep)

                f.write(f"{ep},{reward},{self.epsilon}\n")
                print(f"Ep: {ep}, Reward: {reward}, Lr: {lr:.6e}, Loss: {loss:.4f}, "
                      f"Reward (avg): {avg_reward},  Loss (avg): {avg_loss:.4f}, Epsilon: {self.epsilon}")

                if ep % self.show_every == 0 and self.render:
                    self.env.render()

        self.save(self.model_name)

    def test(self, test_eps=20):
        rewards = []

        for ep in range(test_eps):
            done = False
            state = self.env.reset()
            reward = 0

            while not done:
                action = self.select_actions(state)
                next_state, _, reward, done = self.env.step(action)
                reward += reward
                state = next_state

            rewards.append(reward)

    def select_actions(self, state):
        epsilon = max(self.epsilon, self.epsilon_min)

        if self.random_n != 0:
            self.random_n -= 1
            return self.random_pos - 1
        elif random.random() < epsilon:
            # Return random trading values
            # return [random.uniform(0, 0.1), random.uniform(0, 0.2), random.uniform(0, 0.05)]
            self.random_n = random.sample([1, 3, 5, 9], 1)[0]
            self.random_pos = random.sample([1, 2], 1)[0]
            return self.random_pos - 1

        self.policy_net.eval()
        with torch.no_grad():
            state = torch.Tensor(state).unsqueeze(0)
            out = self.policy_net(state).squeeze()
            self.policy_net.train()
            return torch.argmax(out, dim=0)

    def fill_memory(self):
        for _ in tqdm(range(self.capacity // 4), desc="Initializing replay"):
            state = self.env.reset()
            done = False
            while not done:
                actions = self.select_actions(state)
                next_state, action, reward, done = self.env.step(actions)
                self.replay_mem.store(state, actions, reward, next_state)
                state = next_state

    def update_epsilon(self, ep=0):
        self.epsilon = max(self.epsilon_min, round(self.epsilon * self.decay, 4))

        if ep == 10:
            self.decay = 0.95

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, model_name):
        torch.save(self.policy_net.state_dict(), f"models/{model_name}")

    def load(self, model_name):
        self.policy_net.load_state_dict(torch.load(f"models/{model_name}"))
        self.policy_net.eval()


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.ind = 0

    def store(self, states, actions, rewards, next_states):
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
        indices_to_sample = random.sample(range(len(self.states)), k=batchsize)

        states = np.array(self.states)[indices_to_sample]
        actions = np.array(self.actions)[indices_to_sample]
        next_states = np.array(self.next_states)[indices_to_sample]
        rewards = np.array(self.rewards)[indices_to_sample]

        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.states)

