import numpy as np
import random
import copy
import os
from collections import namedtuple, deque
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from models import Actor, Critic
 
BATCH_SIZE= 200
BUFFER_SIZE= 200000
GAMMA= 0.99
LR_ACTOR= 0.0001
LR_CRITIC= 0.001
TAU=0.001
UPDATE_EVERY= 2
WEIGHT_DECAY= 0.00001

DEVC = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG(object):
    '''
    '''

    def __init__(self, state_size, action_size, nb_agents, rand_seed):
        '''Initialize an MultiAgent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        '''
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,
                                   BATCH_SIZE, rand_seed)
        self.__name__ = 'MADDPG'
        self.nb_agents = nb_agents
        self.na_idx = np.arange(self.nb_agents)
        self.action_size = action_size
        self.act_size = action_size * nb_agents
        self.state_size = state_size * nb_agents
        self.l_agents = [Agent(state_size,
                               action_size,
                               rand_seed,
                               self)
                         for i in range(nb_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        experience = zip(self.l_agents, states, actions, rewards, next_states,
                         dones)
        for i, e in enumerate(experience):
            agent, state, action, reward, next_state, done = e
            na_filtered = self.na_idx[self.na_idx != i]
            others_states = states[na_filtered]
            others_actions = actions[na_filtered]
            others_next_states = next_states[na_filtered]
            agent.step(state, action, reward, next_state, done, others_states,
                       others_actions, others_next_states)

    def act(self, states, add_noise=True):
        na_rtn = np.zeros([self.nb_agents, self.action_size])
        for idx, agent in enumerate(self.l_agents):
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def reset(self):
        for agent in self.l_agents:
            agent.reset()

    def __len__(self):
        return self.nb_agents

    def __getitem__(self, key):
        return self.l_agents[key]


class Agent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, rand_seed, meta_agent):
        '''Initialize an MetaAgent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param rand_seed: int. random seed
        :param memory: ReplayBuffer object.
        '''

        self.action_size = action_size
        self.__name__ = 'DDPG'

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, rand_seed).to(DEVC)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(DEVC)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size,
                                   action_size,
                                   meta_agent.nb_agents,
                                   rand_seed).to(DEVC)
        self.critic_target = Critic(state_size,
                                    action_size,
                                    meta_agent.nb_agents,
                                    rand_seed).to(DEVC)
        # NOTE: the decay corresponds to L2 regularization
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC)  # , weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)

        # Replay memory
        self.memory = meta_agent.memory

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, others_states,
             others_actions, others_next_states):
        self.memory.add(state, action, reward, next_state, done, others_states,
                        others_actions, others_next_states)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # source: Sample a random minibatch of N transitions from R
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        '''Returns actions for given states as per current policy.
        :param states: array_like. current states
        :param add_noise: Boolean. If should add noise to the action
        '''
        states = torch.from_numpy(states).float().to(DEVC)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        '''
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + ? * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        (states, actions, rewards, next_states, dones, others_states,
         others_actions, others_next_states) = experiences
        # rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards
        all_states = torch.cat((states, others_states), dim=1).to(DEVC)
        all_actions = torch.cat((actions, others_actions), dim=1).to(DEVC)
        all_next_states = torch.cat((next_states, others_next_states), dim=1).to(DEVC)

        # --------------------------- update critic ---------------------------
        # Get predicted next-state actions and Q values from target models
        l_all_next_actions = []
        l_all_next_actions.append(self.actor_target(states))
        l_all_next_actions.append(self.actor_target(others_states))
        all_next_actions = torch.cat(l_all_next_actions, dim=1).to(DEVC)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss: L = 1/N SUM{(yi ? Q(si, ai|?Q))^2}
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        # Compute actor loss
        this_actions_pred = self.actor_local(states)
        others_actions_pred = self.actor_local(others_states)
        others_actions_pred = others_actions_pred.detach()
        actions_pred = torch.cat((this_actions_pred, others_actions_pred), dim=1).to(DEVC)
        actor_loss = -self.critic_local(all_states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        # Update the critic target networks
        # Update the actor target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        ?_target = ?*?_local + (1 - ?)*?_target
        :param local_model: PyTorch model. weights will be copied from
        :param target_model: PyTorch model. weights will be copied to
        :param tau: float. interpolation parameter
        '''
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)

    def reset(self):
        self.noise.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x)
        dx += self.sigma * np.random.randn(self.size)  # normal distribution
        self.state = x + dx
        return self.state


class ReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.
        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done",
                                                  "others_states",
                                                  "others_actions",
                                                  "others_next_states"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, others_states,
            others_actions, others_next_states):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done,
                            others_states, others_actions, others_next_states)
        self.memory.append(e)

    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                  if e is not None])).float().to(DEVC)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                   if e is not None])).float().to(DEVC)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                   if e is not None])).float().to(DEVC)
        next_states = torch.from_numpy(np.vstack([e.next_state
                                                  for e in experiences
                                                  if e is not None])).float().to(DEVC)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences
                                            if e is not None]).astype(np.uint8)).float().to(DEVC)

        others_states = torch.from_numpy(np.vstack([e.others_states for e in experiences
                                  if e is not None])).float().to(DEVC)
        others_actions = torch.from_numpy(np.vstack([e.others_actions for e in experiences
                                   if e is not None])).float().to(DEVC)
        others_next_states = torch.from_numpy(np.vstack([e.others_next_states
                                                      for e in experiences
                                                      if e is not None])).float().to(DEVC)

        return (states, actions, rewards, next_states, dones, others_states,
                others_actions, others_next_states)

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)