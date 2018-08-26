import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from prioritized_replay_buffer import PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32        # minibatch size or size of the sampling
GAMMA = 0.99            # discount factor
TAU = 0.5              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the Q network
TRANSFER_EVERY = 4      # how often to transfer from Q to Q_
UPDATE_PRIORITY_EVERY = 32 # how often are we updating the priorities in the memory
DEFAULT_PRIORITY = 1e-10 # how much is the minimum priority given to each experience
PRIORITY_FACTOR = 0.7   # define if the probabilities of sampling would be close to uniform when PRIORITY_FACTOR is close to 0 or 
                        # close to prioritized replay when PRIORITY_FACTOR is close to 1

device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, QNetwork):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            QNetwork: a class inheriting from torch.nn.Module that define the structure of the neural network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        #when using a dropout the qnetwork_target should be put in eval mode
        self.qnetwork_target.eval()
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(seed, device, action_size, BUFFER_SIZE, BATCH_SIZE, DEFAULT_PRIORITY, PRIORITY_FACTOR)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0
        self.up_step = 0
        # To control the importance sampling weight. As the network is converging, b should move toward 1
        self.b = torch.tensor(1., device=device, requires_grad=False)
        self.b_decay = torch.tensor(0.00015, device=device, requires_grad=False)
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int_)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int_)
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % TRANSFER_EVERY
        self.u_step = (self.u_step + 1) % UPDATE_EVERY
        self.up_step = (self.up_step + 1) % UPDATE_PRIORITY_EVERY
        
        # Learn from experiences
        if len(self.memory) > BATCH_SIZE and self.u_step == 0:
            # sample the experiences from the memory based on their priority
            experiences = self.memory.sample()
            self.learn(experiences)
        # Transfer the knowledge from the local network to the fixed on
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        # Update the priorities in the memory to alter the sampling
        # Ideally, this should be done before the sampling is taking place
        # But, for sake of performance, it might be better to recalculate them less often
        if len(self.memory) > 1 and self.up_step == 0:
            for experiences in self.memory.get_all_experiences(512):
                with torch.no_grad():
                    self.qnetwork_local.eval()
                    current_estimate, from_env = self.get_target_estimate(experiences)
                    # update the priorities based on newly learned errors
                    self.memory.update(experiences[-1], (from_env - current_estimate).squeeze())
            
    def get_target_estimate(self, experiences):
            states, actions, rewards, next_states, dones, probabilities, selected = experiences
            with torch.no_grad():
                best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                evaluations = self.qnetwork_target(next_states).gather(1,best_actions)
                from_env = rewards + GAMMA*evaluations*(1 - dones)
            return self.qnetwork_local(states).gather(1, actions), from_env
        
    def learn(self, experiences):
        self.qnetwork_local.train()
        current_estimate,from_env = self.get_target_estimate(experiences)
        probabilities = experiences[-2]
        errors = (from_env - current_estimate)
        # Since the experiences were retrieved based on a given probabilities, such experience will biase the network
        # Therefore, we introduce here an importance sampling weight
        loss = (errors * errors / (len(self.memory) * probabilities) * self.b).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.b = min(self.b + self.b_decay,1)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        θ_target = θ_target + τ*(θ_local - θ_target)

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)