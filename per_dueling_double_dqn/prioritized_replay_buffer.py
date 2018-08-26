import numpy as np
import torch
import random
import itertools

from collections import deque
class Experience:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = float(action)
        self.reward = float(reward)
        self.next_state = next_state
        self.done = float(done)
        self.priority = .0
        self.probability = .0
        
class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, seed, device, action_size, buffer_size, batch_size, default_priority, priority_factor):
        """Initialize a ReplayBuffer object.

        Params
        ======
            seed (int): random seed
            device (string): name of the device to be used by PyTorch
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            default_priority: minimum priority given to the experiences so that the agent will still explore them if necessary
            priority_factor: when close to one, the experiences will be sampled based on only their priorities,
                            when close to zero, it will be a sampling based on normal distribution
        """
        self.seed = np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.priority_factor = priority_factor
        self.default_priority = default_priority
        self.device = device
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        # When adding an experience, its priority is set to the default priority
        # in the main code of the agent, the priorities are not calculated before every sampling but less often
        # that is necessary for performance improvement
        # so when an experience is added, it is first given a default priority which will be updated by the agent afterwards
        e.priority = self.default_priority**self.priority_factor
        self.memory.append(e)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def sample(self):
        """Sample a batch of experiences from memory based on priority."""
        priorities = [e.priority for e in self.memory]
        sum_priorities = np.sum(priorities)
        
        sampling_probs = []
        for exp in self.memory:
            # update the probability of the experience which is as well an importance sampling weight that is necessary
            # during the training of the network
            exp.probability = exp.priority / sum_priorities
            sampling_probs.append(exp.probability)
        experiences = np.random.choice(self.memory, size=self.batch_size, replace=False, p=sampling_probs)
        return self.transform_experiences_to_tensor(experiences)
     
    def transform_experiences_to_tensor(self, experiences):
        # move the experiences to the device being used
        temp_arrays = torch.tensor([[e.state, e.next_state] for e in experiences], device=self.device, dtype=torch.float32).requires_grad_(False)
        states = temp_arrays[:,0].detach()
        next_states = temp_arrays[:,1].detach()
        temp_arrays = torch.tensor([[e.action, e.reward, e.done, e.probability] for e in experiences], device=self.device, dtype=torch.float32).requires_grad_(False)
        actions = temp_arrays[:,0].detach().long().unsqueeze(1)
        rewards = temp_arrays[:,1].detach().unsqueeze(1)
        dones = temp_arrays[:,2].detach().unsqueeze(1)
        probabilities = temp_arrays[:,3].detach().unsqueeze(1)
        
        return (states, actions, rewards, next_states, dones, probabilities, experiences)
    
    def get_all_experiences(self, batch_size):
        # prepare the memory to be swipped by the agent for an update of the priorities
        start=0
        for start in range(0,len(self.memory),batch_size):
            experiences = list(itertools.islice(self.memory, start, start + batch_size))
            yield self.transform_experiences_to_tensor(experiences)
    
    def update(self, experiences, errors):
        # update of the priorities
        for exp, error in zip(experiences, errors.cpu().data.numpy()):
            exp.priority = (np.abs(error.item()) + self.default_priority)**self.priority_factor