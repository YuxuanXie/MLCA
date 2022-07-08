import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.drnn import DRNN

class GBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def put(self, x):
        if len(self.memory) == self.capacity:
            self.memory.pop(0)
        self.memory.append(x)

    def mean(self):
        return sum(self.memory)/ len(self.memory)

class FuNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(FuNAgent, self).__init__()
        self.args = args

        self.f_percept = nn.Linear(input_shape, args.percept_hidden_dim)

        # manager network
        self.Mspace = nn.Linear(args.percept_hidden_dim, args.Msapce_hidden_dim)
        self.Mrnn = nn.GRUCell(args.Msapce_hidden_dim, args.drnn_hidden_dim)
        self.goal = nn.Linear(args.drnn_hidden_dim, args.n_goals)

        # worker 
        self.Wrnn = nn.GRUCell(args.percept_hidden_dim, args.rnn_hidden_dim)
        self.U = nn.Linear(args.rnn_hidden_dim, self.args.W_dim*args.n_actions)
        # linear projection for goal which has no bias
        self.phi = nn.Linear(args.n_goals, args.W_dim, bias=False)

        self.g_buffer = GBuffer(args.n_layers)


    def init_hidden(self):
        # make hidden states on same device as model
        return [ self.Mspace.weight.new(1, self.args.drnn_hidden_dim).zero_() for i in range(self.args.n_layers) ] + [self.f_percept.weight.new(1, self.args.rnn_hidden_dim).zero_()]


    def forward(self, inputs, hidden_states):
        z = F.relu(self.f_percept(inputs))
        s = self.Mspace(z)
        dh_in = hidden_states[0].reshape(-1, self.args.drnn_hidden_dim)
        dh = self.Mrnn(s, dh_in)
        g = self.goal(dh)
        g_norm = torch.norm(g, p=2, dim=1).unsqueeze(1)
        g = g / g_norm.detach()
        self.g_buffer.put(g.detach())

        h_in = hidden_states[1].reshape(-1, self.args.rnn_hidden_dim)
        h = self.Wrnn(z, h_in)
        u = self.U(h)
        w = self.phi(self.g_buffer.mean())
        logits = torch.bmm(u.reshape(self.args.batch_size * self.args.n_agents, self.args.n_actions, self.args.W_dim), w.reshape(self.args.batch_size*self.args.n_agents, self.args.W_dim, 1))
        return g, torch.squeeze(logits), dh, h
