import numpy as np
import random
from collections import namedtuple, deque


import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.5               # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 1        # how often to update the network
TRANSFER_EVERY = 4      # how often to transfer from the online Q network to the targeted fixed Q_ network

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.u_step = 0
        self.t_step = 0
    
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
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int_)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int_)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % TRANSFER_EVERY
        self.u_step = (self.u_step + 1) % UPDATE_EVERY
        
        # if compared to this research paper (https://arxiv.org/abs/1509.06461)
        # the gradient descent should be done on every step and the transfer of local to the target should be done based on self.t_step
        # in our case, the transfer from the online Q Network to the fixed one is done gradually with a rate TAU
        if len(self.memory) > BATCH_SIZE and self.u_step == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # compute the action-values based on the fixed Q Network and the environment
        # that is immediate_rewards + gamma * max(fixed_Q(next_state, actions))
        # these values are stored in the variable from_env

        with torch.no_grad():
            #use qnetwork_local to find the best actions
            #use qnetwork_target to evaluate its action-value
            #from the following, we get the indices of the best actions according to qnetwork_local
            best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            evaluations = self.qnetwork_target(next_states)
            evaluations = evaluations.gather(1,best_actions)
            
            from_env = rewards + gamma*evaluations*(1 - dones)
        # get the action values estimated by the online Q Network
        current_estimate = self.qnetwork_local(states)
        current_estimate = current_estimate.gather(1,actions)
        
        # calculate the loss and backprobagate
        loss = F.mse_loss(current_estimate, from_env)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        θ_target = θ_target + τ*(θ_local - θ_target)
        θ_local = r + gamma * θ_local(s+1)

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        # this is transferring gradually the parameters of the online Q Network to the fixed one
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device).requires_grad_(False)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device).requires_grad_(False)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device).requires_grad_(False)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device).requires_grad_(False)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device).requires_grad_(False)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)