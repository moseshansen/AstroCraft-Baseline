import numpy as np
from scipy.special import softmax

class PromisingStateBuffer():
    def __init__(self, buffer_size, obs_size, action_space_size):
        """Initializes a buffer with buffer_size rows and obs_size +1 columns"""
        self.buffer = np.zeros((buffer_size, obs_size+1))
        self.action_dict = {i:[] for i in range(buffer_size)}
        self.action_space_size = action_space_size
        self.buffer_size = int(buffer_size)
        self.action_counts = np.ones(action_space_size)
        self.state_dist = np.ones(buffer_size) / buffer_size

    def push(self, state, actions, score):
        """Pushes a new start state to the buffer if its score is better than at least one of the states in the buffer"""
        worst_idx = np.argmin(self.buffer[:,-1])
        worst_score = self.buffer[worst_idx,-1]
        if score < worst_score:
            return False

        new_row = np.concatenate((state, np.array([score])))
        self.buffer[worst_idx] = new_row
        self.action_dict[worst_idx] = actions
        self._calculate_dist_actions()
        self._calculate_dist_states()
        return True
    
    def remove(self, n):
        """Removes the state at row n"""
        self.buffer[n,:] = 0
        self.action_dict[n] = []
        self._calculate_dist_actions()
        self._calculate_dist_states()
    
    def _calculate_dist_actions(self):
        """Calculates a probability distribution for the actions"""
        # Tabulate action frequencies across all states
        all_arrays = sum(list(self.action_dict.values()), [])
        flat_array = np.array(all_arrays)
        unique_values, value_counts = np.unique(flat_array, return_counts=True)

        self.action_counts = np.zeros(self.action_space_size)
        self.action_counts[unique_values] = value_counts

    def _calculate_dist_states(self):
        """Calculates a probability distribution for the states by softmaxing their scores"""
        # Get only the rows that have a score gt 0
        valid_states = np.where(self.buffer[:,-1] > 0)

        # Softmax the scores in those rows
        softmaxes = softmax(self.buffer[valid_states,-1])

        # Update state dist
        self.state_dist = np.zeros_like(self.state_dist)
        self.state_dist[valid_states] = softmaxes

    def sample_action(self, action_mask):
        """Given an action mask for a single mobile agent, samples from the stored probability distribution"""
        valid_idx_list = np.where(action_mask == 1)[0].tolist()
        random_draw = all(self.action_counts[i] == 0 for i in valid_idx_list)

        # If none of the valid actions have been observed previously, pick a random valid action
        if random_draw:
            return np.random.choice(valid_idx_list)

        masked_vals = self.action_counts * action_mask
        dist = masked_vals / np.sum(masked_vals)
        return np.random.choice(self.action_space_size, p=dist)
    
    def sample_state(self):
        """Samples a state from the buffer with probability given by the softmaxed score column"""
        idx = np.random.choice(self.buffer_size, p=self.state_dist)
        row = self.buffer[idx,:-1]
        return row
    
    def __str__(self):
        """Prints the buffer"""
        return str(self.buffer)