import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from encoder import Encoder
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class DQTN(nn.Module):
    def __init__(self, dims, lr, dropout, embeddings, layers, heads, fwex, neurons):
        super(DQTN, self).__init__()

        self.dims = dims
        self.lr = lr
        self.dropout = dropout

        self.encoder = Encoder(embeddings, layers, heads, fwex, dropout, 186)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons // 8)
        self.fc3 = nn.Linear(neurons // 8, neurons // 8)
        self.fc4 = nn.Linear(neurons // 8, 2)
        self.softmax = nn.Softmax(dim=1)

        self.loss = None
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, step_size=10, gamma=0.75)

    def forward(self, x):
        x = self.flat(self.encoder(x))
        x = F.dropout(F.relu(self.fc1(x)), self.dropout)
        x = F.dropout(F.relu(self.fc2(x)), self.dropout)
        x = F.dropout(F.relu(self.fc3(x)), self.dropout)
        x = self.softmax(self.fc4(x))
        # scale_factors = torch.tensor([0.1, 0.2, 0.05], device=x.device)
        return x

    def backward(self, y_pred, y_targ):
        self.optimizer.zero_grad()
        self.loss = F.cross_entropy(y_pred, y_targ)
        self.loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        # self.scheduler.step()