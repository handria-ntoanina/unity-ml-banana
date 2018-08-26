import numpy as np
import torch
class ReplayTensorBuffer:
    """Fixed-size buffer to store experience tuples straight in pytorch device"""

    def __init__(self, buffer_size, state_size, batch_size, default_prob, priority_factor):
        """Initialize a ReplayTensorBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.position = 0
        self.priority_factor = priority_factor
        self.default_prob = default_prob
        self.states = torch.zeros([buffer_size, state_size], dtype=torch.float32, requires_grad=False).to(device)
        self.actions = torch.zeros([buffer_size], dtype=torch.int64, requires_grad=False).to(device)
        self.rewards = torch.zeros([buffer_size], dtype=torch.float32, requires_grad=False).to(device)
        self.next_states = torch.zeros([buffer_size, state_size], dtype=torch.float32, requires_grad=False).to(device)
        self.dones = torch.zeros([buffer_size], dtype=torch.float32, requires_grad=False).to(device)
        self.priorities = torch.zeros([buffer_size], dtype=torch.float32, requires_grad=False).to(device)
    def add(self, state, action, reward, next_state, done, error):
        """Saves a transition."""
        self.states[self.position] = torch.from_numpy(state).type(torch.float32)
        self.actions[self.position] = action.item()
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.from_numpy(next_state).type(torch.float32)
        self.dones[self.position] = float(done)
        priorities = (abs(error) + self.default_prob)**self.priority_factor
        self.priorities[self.position] = priorities
        self.position = (self.position + 1) % self.buffer_size
    def update(self, selected, errors):
        """
        Params
        ======
            state, action, and error are batches
        """
        self.priorities[selected] = (errors.abs() + self.default_prob)**self.priority_factor
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # TO FINISH: build the probabilities using the network
        sampling_probs = self.priorities / self.priorities.sum()
        # should use multinomial instead
        selected = torch.multinomial(sampling_probs, self.batch_size, replacement=False)
        states = self.states[selected].unsqueeze(1)
        actions = self.actions[selected].unsqueeze(1)
        rewards = self.rewards[selected].unsqueeze(1)
        next_states = self.next_states[selected].unsqueeze(1)
        dones = self.dones[selected].unsqueeze(1)
        sampling_probs = sampling_probs[selected]
        return (states, actions, rewards, next_states, dones, sampling_probs, selected)

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.position+1,self.buffer_size)