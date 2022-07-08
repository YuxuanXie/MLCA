import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.latent import LatentCritic 
from utils.rl_utils import build_td_lambda_targets
from components.action_selectors import multinomial_entropy, multinomial_gumbel_entropy
import torch as th
from torch.optim import RMSprop, Adam


class FuNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.worker_critic = LatentCritic(scheme, args, worker=True)
        self.target_worker_critic = copy.deepcopy(self.worker_critic)

        self.manager_critic = LatentCritic(scheme, args, worker=False)
        self.target_manager_critic = copy.deepcopy(self.manager_critic)

        self.agent_params = list(mac.parameters())
        self.critic_params = list(self.worker_critic.parameters()) + list(self.manager_critic.parameters())
        self.params = self.agent_params + self.critic_params
        
        self.entropy_coef = args.entropy_coef
        
        if self.args.adam:
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)
            self.critics_optimiser = Adam(params=self.critic_params, lr=args.critic_lr)
        else :
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
            self.critics_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        bs = batch.batch_size
        max_t = batch.max_seq_length
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # mask = mask.repeat(1, 1, self.n_agents).view(-1)

        mac_out = []
        mac_out_goals = []
        mac_out_entropy = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            goals, agent_outs = self.mac.forward(batch, t=t)
            if self.args.action_selector == "gumbel":
                agent_out_entropy = multinomial_gumbel_entropy(probs=agent_outs).mean(dim=-1, keepdim=True)
            else:
                agent_out_entropy = multinomial_entropy(probs=agent_outs).mean(dim=-1, keepdim=True)
            if th.isnan(agent_out_entropy).any():
                import pdb; pdb.set_trace()
            mac_out.append(agent_outs)
            mac_out_goals.append(goals)
            mac_out_entropy.append(agent_out_entropy)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_out_entropy = th.stack(mac_out_entropy, dim=1)
        mac_out_goals = th.stack(mac_out_goals, dim=1)

        # worker loss
        worker_q_val = self.worker_critic(batch["state"][:,:-1], mac_out)
        worker_loss = (worker_q_val * mask).sum() / mask.sum()

        # manager loss
        manager_mask = copy.deepcopy(mask)
        manager_q_val = self.manager_critic(batch["state"][:,:-1], mac_out_goals)
        manager_loss = (manager_q_val * manager_mask).sum() / manager_mask.sum()

        # Calculate mean entropy
        entropy_mask = copy.deepcopy(mask)
        entropy_loss = (mac_out_entropy * entropy_mask).sum() / entropy_mask.sum()
        entropy_ratio = self.entropy_coef / entropy_loss.item()

        total_loss = - worker_loss - manager_loss - entropy_ratio * entropy_loss
        # total_loss = - (q_val * mask).sum() / mask.sum() - self.entropy_coef * entropy_loss

        # Optimise agents
        self.agent_optimiser.zero_grad()
        total_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        _, running_log = self._train_critic(batch, t_env, episode_num)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            ts_logged = len(running_log["worker_critic_loss"])
            for key in ["worker_critic_loss", "manager_critic_loss", "critic_grad_norm", "td_error_abs", "worker_target_mean", "manager_target_mean"]:
                self.logger.log_stat(key, sum(running_log[key])/ts_logged, t_env)

            # self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("worker_q_t_mean", ((worker_q_val * mask).sum() / mask.sum()).item(), t_env)
            self.logger.log_stat("manager_q_t_mean", ((manager_q_val * manager_mask).sum() / manager_mask.sum()).item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("worker_pi_max", (mac_out.view(bs,max_t-1,-1).max(keepdim=True,dim=2)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("manager_pi_max", (mac_out_goals.view(bs,max_t-1,-1).max(keepdim=True,dim=2)[0] * manager_mask).sum().item() / manager_mask.sum().item(), t_env)
            self.logger.log_stat("entropy", entropy_loss.item(), t_env)
            self.logger.log_stat("total_loss", total_loss.item(), t_env)
            self.logger.log_stat("training_iteration", self.critic_training_steps, t_env)
            self.log_stats_t = t_env


    def _train_critic(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # batch, rewards, terminated, actions, avail_actions, mask, bs, max_t, self, 
        bs = batch.batch_size
        max_t = batch.max_seq_length
        rewards = batch["reward"][:, :-1]
        actions = batch["actions_onehot"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        goals = batch["goals"][:, :]

        # Optimise critic   
        target_worker_q_vals = self.target_worker_critic(batch["state"], actions)[:, :]
        target_manager_q_vals = self.target_manager_critic(batch["state"], goals)[:, :]

        # Calculate td-lambda targets
        worker_targets = build_td_lambda_targets(rewards, terminated, mask, target_worker_q_vals, self.n_agents, self.args.gamma, self.args.td_lambda)
        manager_targets = build_td_lambda_targets(rewards, terminated, mask, target_manager_q_vals, self.n_agents, self.args.gamma, self.args.td_lambda)

        q_vals = th.zeros_like(target_worker_q_vals)[:, :-1]

        running_log = {
            "worker_critic_loss": [],
            "manager_critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "worker_target_mean": [],
            "manager_target_mean": [],
        }

        worker_q = self.worker_critic(batch["state"][:, :-1], actions[:,:-1])
        manager_q = self.manager_critic(batch["state"][:, :-1], goals[:,:-1])

        worker_q_td_error = (worker_q - worker_targets.detach())
        manager_q_td_error = (manager_q - manager_targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = worker_q_td_error * mask

        manager_mask = copy.deepcopy(mask)
        manager_q_td_error = manager_q_td_error * manager_mask

        # Normal L2 loss, take mean over actual data
        worker_loss = (masked_td_error ** 2).sum() / mask.sum()
        manager_loss = (manager_q_td_error ** 2).sum() / manager_mask.sum()

        loss = worker_loss + manager_loss
        
        self.critics_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critics_optimiser.step()
        self.critic_training_steps += 1

        running_log["worker_critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(loss.abs().item())
        running_log["worker_target_mean"].append((worker_targets * mask).sum().item() / mask_elems)
        running_log["manager_target_mean"].append((manager_targets * manager_mask).sum().item() / mask_elems)

        if (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_step = self.critic_training_steps

        return q_vals, running_log

    def _update_targets(self):
        self.target_worker_critic.load_state_dict(self.worker_critic.state_dict())
        self.target_manager_critic.load_state_dict(self.manager_critic.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.worker_critic.cuda()
        self.target_worker_critic.cuda()
        self.manager_critic.cuda()
        self.target_manager_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.worker_critic.state_dict(), "{}/worker_critic.th".format(path))
        th.save(self.manager_critic.state_dict(), "{}/manager_critic.th".format(path))

        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critics_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        
        self.worker_critic.load_state_dict(th.load("{}/worker_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.manager_critic.load_state_dict(th.load("{}/manager_critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_worker_critic.load_state_dict(self.worker_critic.state_dict())
        self.target_manager_critic.load_state_dict(self.manager_critic.state_dict())

        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critics_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

    def get_gaes(self, rewards, v_preds, next_v_preds):
        delta = rewards + self.args.gamma * next_v_preds - v_preds
        gaes = delta.clone()
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.args.td_lambda * self.args.gamma * gaes[t + 1]
        return gaes
