from AstroCraft.PettingZoo_MA.env.CaptureTheFlagMA import CTFENVMA

import numpy as np

class CTFENVMA_sb3(CTFENVMA):
    """
    A wrapper for the CTFENVMA environment that allows
    the environment to be used with sb3's MaskablePPO
    """
    
    def __init__(self, team_size, win_rew, flag_rew):
        super().__init__(team_size, win_rew, flag_rew)
        
        self.obs_space = self.observation_space(0)
        self.act_space = self.action_space(0)
        
    def reset(self, seed=None, options=None):
        """
        Changes the reset method to return just the obs
        and info for player 0, for use with sb3
        """
        obs, info = super().reset(seed, options)
        
        return obs['player0']['observation'], info['player0']
    
    def step(self, action):
        """
        Modifies the step method to only take an action 
        for player 0. Player 1 will take no actions. 
        Action masks are not returned at the end.
        """
        joint_action = {
            'player0': action,
            'player1': np.zeros_like(action)
        }
        
        obs, rew, term, trunc, info = super().step(joint_action)
        return obs['player0']['observation'], rew['player0'], term['player0'], trunc['player0'], info['player0']
        
    def action_mask(self):
        """
        Returns the action mask for player 0 only. If
        there are multiple such masks (e.g. for NvN games)
        the masks are stacked.
        """
        p0_mask, p1_mask = super().valid_action_masks()
        return np.hstack(p0_mask)