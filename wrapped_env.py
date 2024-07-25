from AstroCraft.PettingZoo_MA.env.CaptureTheFlagMA import CTFENVMA, MAX_FUEL
from scripted_agents.dormant_agent import DormantOpponent

from gymnasium import spaces
import numpy as np

class CTFENVMA_sb3(CTFENVMA):
    """
    A wrapper for the CTFENVMA environment that allows
    the environment to be used with sb3's MaskablePPO
    """
    
    def __init__(self, team_size, win_rew, flag_rew):
        super().__init__(team_size, win_rew, flag_rew)
        
        self.action_space = spaces.MultiDiscrete(np.array([self._num_actions]*self._team_size, dtype=int))
        self.observation_space = spaces.Box(low=-3, high=MAX_FUEL, shape=(self._team_size*2+2, 9))
        self.opponent = DormantOpponent(team_size)
        
    def reset(self, seed=None, options=None):
        """
        Changes the reset method to return just the obs
        and info for player 0, for use with sb3
        """
        obs, info = super().reset(seed, options)
        self.obs = obs
        
        return obs['player0']['observation'], info['player0']
    
    def step(self, action):
        """
        Modifies the step method to only take an action 
        for player 0. Player 1 will take no actions. 
        Action masks are not returned at the end.
        """
        joint_action = {
            'player0': action,
            'player1': self.opponent.select_action(self.obs['player1'],1,1,1,1)
        }
        
        obs, rew, term, trunc, info = super().step(joint_action)
        self.obs = obs
        return obs['player0']['observation'], rew['player0'], term['player0'], trunc['player0'], info['player0']
        
    def action_mask(self):
        """
        Returns the action mask for player 0 only. If
        there are multiple such masks (e.g. for NvN games)
        the masks are stacked.
        """
        p0_mask, p1_mask = super().valid_action_masks()
        return np.hstack(p0_mask)