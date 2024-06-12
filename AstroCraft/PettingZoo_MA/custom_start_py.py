"""
A Wrapper for the base astrocraft multi-agent game that will initialize the game in a random state with a few specifics:
-1v1
-player agent is not flagged
-player agent is within a certain distance of the enemy base and thus could capture the flag within a few turns
-only allow game to last a few turns at most
"""

from env.CaptureTheFlagMA import CTFENVMA

from env.moving_object import MovingObject
from env.mobile_agent import MobileAgent
from env.util import MAX_FUEL, MIN_FUEL_DICT, PTMAX
import numpy as np
import random


class CustomStart(CTFENVMA):

    def __init__(self, p0_base_angle=0, p0_agent_angle=0, p0_agent_orbital=0, p0_agent_flagged=0, p1_agent_angle=np.pi, p1_agent_orbital=0, p1_agent_flagged=0, max_fuel=1e3):
        super().__init__(1,1,0)
        self.p0_base_angle = p0_base_angle
        self.p0_agent_angle = p0_agent_angle
        self.p0_agent_orbital = p0_agent_orbital
        self.p0_agent_flagged = p0_agent_flagged
        self.p1_agent_angle = p1_agent_angle
        self.p1_agent_orbital = p1_agent_orbital
        self.p1_agent_flagged = p1_agent_flagged
        self.max_fuel = max_fuel

    def reset(self):
        """Resets the environment in a way consistent with the one step to capture methodology"""
        self._time = 0 # timestep counter for the game

        # Initialize bases, pick a random angle for the opponent's base and shift the player's by pi from that angle
        self._player0[0] = MovingObject(
            None, None, 0, self.p0_base_angle, isbase=True, orbital=0, num=-1)
        self._player1[0] = MovingObject(
            None, None, 1, (self.p0_base_angle + np.pi) % (2*np.pi), isbase=True, orbital=0, num=-1)

        # create MobileAgent objects
        # For player0, select a random orbital (not geo), and find an angle such that the agent is within a certain distance of the target
        # Infinite fuel
        self._player0[1] = MobileAgent(
            None, None, 0, 1, True, self.max_fuel, self.p0_agent_flagged, False, self.p0_agent_orbital, self.p0_agent_angle, isbase=False, verbose=False)
        self._player1[1] = MobileAgent(
            None, None, 1, 1, True, 1e10, self.p1_agent_flagged, False, self.p1_agent_orbital, self.p1_agent_angle, isbase=False, verbose=False)
        
        # Get obs for both agents
        p0_obs = self.get_obs('player0')
        p1_obs = self.get_obs('player1')

        # Create masks for action space
        p0_mask, p1_mask = self.valid_action_masks()

        obs = {'player0':{'observation': p0_obs, 'action_mask':p0_mask}, 'player1':{'observation':p1_obs, 'action_mask':p1_mask}}

        return obs, {"player0":{}, "player1":{}}
    
    def reward(self, agent):
        """
        This method assigns the reward per timestep dependent on the state of the system. It also determines whether the game's done state is True.
        The implemented reward structure is:
        :rtype: int, bool
        """
        if agent == "player0":
            agent = self._player0
            opponent = self._player1
        else:
            agent = self._player1
            opponent = self._player0

        terminated = False
        truncated = False
        reward = 0

        # Check if agent has at least one active mobile
        all_dead_agent = False
        all_dead_opponent = False
        if all([x._fuel <= MIN_FUEL_DICT[x._orbital] or not x._alive and not x._transferring for x in agent]):
            all_dead_agent = True

        if all([x._fuel <= MIN_FUEL_DICT[x._orbital] or not x._alive and not x._transferring for x in opponent]):
            all_dead_opponent = True

        # If neither the agent nor the opponent has any remaining active mobiles
        elif all_dead_agent and all_dead_opponent:
            truncated = True
        
        # Get rewards of events due to collisions
        for i in range(self._team_size):

            # If the agent's mobile returns the flag to base
            if agent[i+1]._returned_flag:
                #print("Game Over: player0 wins!")
                terminated = True
                reward += self._win_rew

            # If the opponent's mobile returns the flag to base
            elif opponent[i+1]._returned_flag:
                #print("Game Over: player1 wins!")
                terminated = True
                reward -= self._win_rew
    
        # If time is up
        if self._time >= PTMAX:  
            #print("Game Over: draw. Time limit elapsed.")
            truncated = True

        return reward, terminated, truncated