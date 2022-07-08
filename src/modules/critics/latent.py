import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LatentCritic(nn.Module):
    def __init__(self, scheme, args, worker=True):
        super(LatentCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions if worker else args.n_goals
            
        self.n_agents = args.n_agents

        self.state_dim = scheme["state"]["vshape"]
        self.output_type = "q"

        self.embed_dim = args.mixing_embed_dim * self.n_agents * self.n_actions
        self.hid_dim = args.mixing_embed_dim

 
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim))
        self.hyper_b_1 = nn.Linear(self.state_dim, self.hid_dim)

        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hid_dim, self.hid_dim))
        self.hyper_b_final = nn.Sequential(nn.Linear(self.state_dim, self.hid_dim),
                               nn.ReLU(),
                               nn.Linear(self.hid_dim, 1))


    def forward(self, state, policy, t=None):
        bs = state.shape[0]
        max_t = state.shape[1] if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)

        state = state[:, ts]
        probs = policy[:, ts].reshape(-1, 1, self.n_agents * self.n_actions)

        w1 = self.hyper_w_1(state)
        b1 = self.hyper_b_1(state)    
        

        w1 = w1.view(-1, self.n_agents * self.n_actions, self.hid_dim)
        b1 = b1.view(-1, 1, self.hid_dim)

        h = F.relu(th.bmm(probs, w1) + b1)

        w_final = self.hyper_w_final(state)
        b_final = self.hyper_b_final(state)

        w_final = w_final.view(-1, self.hid_dim, 1)
        b_final = b_final.view(-1, 1, 1)

        q = th.bmm(h, w_final) + b_final
 
        return q.view(bs, -1, 1)

    # def _build_inputs(self, state, policy, t=None):
    #     bs = state.shape[0]
    #     max_t = state.shape[1] if t is None else 1
    #     ts = slice(None) if t is None else slice(t, t+1)
    #     inputs = []
    #     # state
    #     # inputs.append(state[:, ts])

    #     # observation
    #     # inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

    #     # actions 
    #     inputs.append(policy[:, ts].view(bs, max_t, -1))

    #     inputs = th.cat([x.reshape(bs, max_t, -1) for x in inputs], dim=-1)
    #     return inputs

    # def _get_input_shape(self, scheme):
    #     input_shape = 0
    #     # state
    #     # input_shape += scheme["state"]["vshape"]
    #     # observation
    #     # input_shape += scheme["obs"]["vshape"] * self.n_agents
    #     # actions
    #     input_shape = scheme["actions_onehot"]["vshape"][0] * self.n_agents
    #     return input_shape
