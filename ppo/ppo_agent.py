import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from ppo.ppo import PPOActorCritic  # your ResNet‐based actor‐critic with GRU

class RolloutBuffer:
    """
    On‐policy buffer for storing one episode (or rollout) of transitions.
    After each episode, we consume these to do a PPO update.
    """
    def __init__(self):
        self.states       = []
        self.actions      = []
        self.logprobs     = []
        self.values       = []
        self.rewards      = []
        self.is_terminals = []
        self.valid_actions   = []

    def clear(self):
        """Empty the buffer in place."""
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.values[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.valid_actions[:]


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent with GRU memory.
    - Uses clipped surrogate objective.
    - Shares a common ResNet trunk + GRU, with separate policy & value heads.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        valid_coef: float = 0.1, 
        max_grad_norm: float = 0.5,
        custom_flag: bool = False
    ):
        """
        Args:
          state_dim:    tuple, e.g. (H, W) from env.observation_space
          action_dim:   int, number of discrete actions
          device:       torch.device
          lr:           learning rate
          gamma:        discount factor
          eps_clip:     PPO clipping ε
          k_epochs:     how many passes over each collected rollout
          ent_coef:     weight for entropy bonus
          vf_coef:      weight for value‐loss term
          max_grad_norm:gradient clipping threshold
          custom_flag:  if True, swap in a graph‐based policy
        """
        self.device       = device
        self.gamma        = gamma
        self.eps_clip     = eps_clip
        self.k_epochs     = k_epochs
        self.ent_coef     = ent_coef
        self.vf_coef      = vf_coef
        self.max_grad_norm= max_grad_norm
        self.valid_coef = valid_coef

        # GRU hidden states (actor & critic)
        self.hx_actor   = None
        self.hx_critic  = None

        # 1) build actor‐critic networks
        self.policy      = PPOActorCritic(action_dim).to(device)
        self.policy_old  = PPOActorCritic(action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 2) optimizer only on the "new" policy
        self.optimizer   = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        # 3) on‐policy rollout buffer
        self.buffer      = RolloutBuffer()

    def reset_memory(self):
        """
        Call at the start of each new episode to zero out the GRU states.
        """
        self.hx_actor, self.hx_critic = None, None

    def select_action(self, state, valid_actions, mask_prob: float = 1.0):
        """
        Given a single observation, select an action using the old policy.
        Stores (state, action, logprob, value) in the rollout buffer.
        Returns: action as Python int.
        """
        # ─── 1.  preprocess single obs → batched tensor dict ──────────────
        if isinstance(state, dict):
            grid      = torch.FloatTensor(state['grid']).unsqueeze(0).to(self.device)
            direction = torch.FloatTensor(state['direction']).unsqueeze(0).to(self.device)
            distance  = torch.FloatTensor(state['distance']).unsqueeze(0).to(self.device)
            obs = {'grid': grid, 'direction': direction, 'distance': distance}
        else:
            obs = state                                  # already a dict of tensors

        # ─── 2.  forward through frozen old policy ────────────────
        with torch.no_grad():
            out = self.policy_old(obs, self.hx_actor, self.hx_critic)
            if len(out) == 2:
                logits, value       = out
                new_hx_a, new_hx_c  = None, None
            else:
                logits, value, new_hx_a, new_hx_c = out                 # (1,5)

            # ── 2.a  action-masking ─────────────────────────────────
            logits = logits.squeeze(0)                                  # (5,)
            # 2.a ---- optionally mask invalid actions -------------------------
            if random.random() < mask_prob:            # ~30 % chance
                mask = torch.full_like(logits, float('-inf'))
                mask[valid_actions] = 0.0              # keep only valid ones
                logits = logits + mask                 # masked logits

            probs   = F.softmax(logits, dim=-1).clamp(min=1e-8)
            dist    = torch.distributions.Categorical(probs)
            action  = dist.sample()                                     # tensor []
            logprob = dist.log_prob(action)

        # ─── 3.  carry forward GRU memory ──────────────────────────
        self.hx_actor, self.hx_critic = new_hx_a, new_hx_c

        # ─── 4.  save to rollout buffer ────────────────────────────
        self.buffer.states.append(obs)
        self.buffer.actions.append(action.unsqueeze(0))
        self.buffer.logprobs.append(logprob.unsqueeze(0))
        self.buffer.values.append(value)
        self.buffer.valid_actions.append(valid_actions)

        # print(valid_actions, action.item())
        return action.item()

    def push_reward(self, reward, done):
        """
        Append the immediate reward and terminal flag for the last taken action.
        Call right after env.step(). 
        """
        self.buffer.rewards.append(torch.tensor(
            reward, dtype=torch.float32, device=self.device))
        self.buffer.is_terminals.append(done)

    def update(self):
        """
        Perform a PPO update using the transitions in the rollout buffer.
        Unrolls the entire episode through the GRU for BPTT.
        """
        # nothing to do if we never collected any rewards
        if len(self.buffer.rewards) == 0:
            return 0.0

        # 1) compute discounted returns
        returns = []
        discounted = 0
        for reward, is_term in zip(reversed(self.buffer.rewards),
                                   reversed(self.buffer.is_terminals)):
            if is_term:
                discounted = 0
            discounted = reward + self.gamma * discounted
            returns.insert(0, discounted)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # 2) convert stored values
        values = torch.cat(self.buffer.values).squeeze(-1)  # (T,)

        # 3) compute advantages (GAE-0)
        advantages = returns - values
        adv_mean, adv_std = advantages.mean(), advantages.std(unbiased=False)
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        # 4) old logprobs & actions
        old_logprobs = torch.cat(self.buffer.logprobs).detach()
        actions      = torch.cat(self.buffer.actions)

        all_losses = []
        # raw = {'policy':[], 'value':[], 'entropy':[], 'valid':[]}
        # 5) PPO epochs with full unroll through GRU
        T = len(returns)
        for _ in range(self.k_epochs):
            hx_a, hx_c = None, None
            for t in range(T):
                # reset memory at episode boundaries
                if self.buffer.is_terminals[t]:
                    hx_a, hx_c = None, None

                obs_t     = self.buffer.states[t]
                ret_t     = returns[t]
                adv_t     = advantages[t]
                old_lp_t  = old_logprobs[t]
                act_t     = actions[t]
                valid    = self.buffer.valid_actions[t]

                # forward through current policy
                out = self.policy(obs_t, hx_a, hx_c)
                if len(out) == 2:
                    logits_t, val_t = out
                    new_hx_a, new_hx_c = None, None
                else:
                    logits_t, val_t, new_hx_a, new_hx_c = out

                # carry forward hidden state
                hx_a, hx_c = new_hx_a, new_hx_c

                # compute losses for this time‐step
                probs_t      = F.softmax(logits_t, dim=-1).clamp(min=1e-8)
                dist_t       = torch.distributions.Categorical(probs_t)
                new_logprob  = dist_t.log_prob(act_t)
                entropy_t    = dist_t.entropy().mean()

                ratio        = (new_logprob - old_lp_t).exp()
                surr1        = ratio * adv_t
                surr2        = torch.clamp(ratio,
                                           1.0 - self.eps_clip,
                                           1.0 + self.eps_clip) * adv_t
                policy_loss  = -torch.min(surr1, surr2)

                value_loss   = F.mse_loss(val_t, ret_t.unsqueeze(0))

                # 3) *new* valid‐action loss (binary‐cross­entropy on the logits)
                #    build a length-A mask (1 for valid, 0 for invalid)
                A = logits_t.size(-1)
                mask = logits_t.new_zeros(A)
                mask[valid] = 1.0
                valid_loss = F.binary_cross_entropy_with_logits(
                    logits_t.squeeze(0), mask
                )

                all_losses.append(policy_loss + self.vf_coef * value_loss
                                  - self.ent_coef * entropy_t + self.valid_coef * valid_loss)
                
                # raw['policy'].append(policy_loss.item())
                # raw['value'].append(value_loss.item())
                # raw['entropy'].append(entropy_t.item())
                # raw['valid'].append(valid_loss.item())

        # 6) back‐propagate through the whole unrolled sequence
        loss = torch.stack(all_losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 7) sync old policy & clear buffer & reset memory
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        self.reset_memory()

        # for k,v in raw.items():
        #     print(f"  avg_{k} = {np.mean(v):.4f}")

        return loss.item()

    def save(self, path):
        """Save the PPO policy + optimizer state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """Load saved policy + optimizer state into both policy & old policy."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.policy_old.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
