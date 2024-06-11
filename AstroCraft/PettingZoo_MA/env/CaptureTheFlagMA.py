from pettingzoo.utils.env import ParallelEnv
import gymnasium as gym
import cv2
from gymnasium import spaces
import matplotlib.pyplot as plt
import pickle
from .moving_object import MovingObject
from .mobile_agent import MobileAgent
from collections import defaultdict
from .constants import MAX_FUEL, PTACT, PCUBE, PTCOL, PTMAX, PRAD, R0, DH, DHACTUAL, MIN_FUEL_DICT, SLOW_FUEL_COSTS, FAST_FUEL_COSTS
from .util import check_collisions, check_intercept
import numpy as np
import math
from scipy.linalg import norm
from functools import lru_cache


class CTFENVMA(ParallelEnv):
    metadata = {
        "name": "CaptureTheFlagMA",
    }

    def __init__(self, team_size, win_rew, flag_rew):
        self._team_size = team_size
        self._win_rew = win_rew
        self._flag_rew = flag_rew
        self._num_orbitals = 7
        self._player0 = [None]*(self._team_size+1) # Objects populated in reset()
        self._player1 = [None]*(self._team_size+1) # Objects populated in reset()
        self.agents = ['player0','player1']
        self.possible_agents = self.agents.copy()
        self._verbose = False

        self._num_actions = 12 + 2*team_size
        # self.action_space = spaces.MultiDiscrete(np.array([self._num_actions]*self._team_size, dtype=np.int))

        # self.observation_space = spaces.Box(low=-3, high=MAX_FUEL, shape=(self._team_size*2+2, 8))

    def reset(self, seed=None, options=None):
        """
        Resets environment to starting state.
        This function should: 
        
        | - Instantiate the teams in play (there are always two teams), and choose the starting point of the two bases (preset, random, or user defined)
        | - Reset the reward variables
        | - All agents are UNFLAGGED
        | - All agents are PRESENT/ALIVE
        | - Fuel is set to maximum value
        | - No mobile agent is TRANSFERRING
        | - Only orbital 0 is occupied
        Note: The number of teams in any given game is always two. If the starting state
        is set to random, only one team's base agent should be randomly placed. The other
        should instantiate 180 degrees away.
        :returns: the starting state of the game, by calling get_obs(), and a blank information dictionary called info
        :rtype: np.ndarray
        """
        #super().reset(seed=seed) #Seeding according to Gymnasium documentation
        self._time = 0 # timestep counter for the game

        # Initialize bases
        self._player0[0] = MovingObject(
            None, None, 0, 0, isbase=True, orbital=0, num=-1)
        self._player1[0] = MovingObject(
            None, None, 1, np.pi, isbase=True, orbital=0, num=-1)

        # create MobileAgent objects
        dim = self._team_size+1
        for i in range(1, dim):
            self._player0[i] = MobileAgent(
                None, None, 0, i, True, MAX_FUEL, False, False, 0, 0, isbase=False, verbose=False)
            self._player1[i] = MobileAgent(
                None, None, 1, i, True, MAX_FUEL, False, False, 0, np.pi, isbase=False, verbose=False)
        
        # Get obs for both agents
        p0_obs = self.get_obs('player0')
        p1_obs = self.get_obs('player1')

        # Create masks for action space
        p0_mask, p1_mask = self.valid_action_masks()

        obs = {'player0':{'observation': p0_obs, 'action_mask':p0_mask}, 'player1':{'observation':p1_obs, 'action_mask':p1_mask}}

        return obs, {"player0":{}, "player1":{}}

    def get_obs(self, agent):
        '''
        The purpose of this function is to collect observations (state space) from both 
        players' MovingObject instances and return a 2D array of all the attribute data.
        The format is agent's MovingObjects concatenated with opponent's MovingObjects
        A MovingObject state is defined as [team, num, alive, fuel, flag, transferring, orbital, angle] and is called by the get_state() method.
        If noisy, the observation is skewed by a distribution.
        
        :returns: an array of serialized MovingObject data
        :rtype: np.ndarray
        '''

        if agent == "player0":
            agent = self._player0
            opponent = self._player1
        else:
            agent = self._player1
            opponent = self._player0

        obs = np.zeros((self._team_size*2+2, 9)).astype(np.float32)

        for i in range(self._team_size+1):
            # Adding +1 to account for the base station
            obs[i] = agent[i].get_state()
            obs[self._team_size + i + 1] = opponent[i].get_state()

        return obs

    def step(self, actions):
        """
        This method implements the motion of all mobile agents over the span of a single timestep.
        Though bases cannot take actions, they are included in the action space to keep a similar dimension as the state space. #! unclear if this is good or bad
        The controller specifies an array of size team_size+1, with an integer action corresponding to every MovingObject instance.
        The step function takes in this array, and concatenates the actions of the opponent's MovingObject instances.
        Thus, all MovingObject instances step forward with some action every time the environment step function is called.
        | Goes through a single step of the environment and moves all agents based on action inputs.
        | Action space (for NvN where n ranges from 1 to N):

        0:      Continue current transfer trajectory (should be only valid move while transferring, masked if not transferring)
        
        1:      Jump to or Continue along Orbital -3 
        2:      Jump to or Continue along Orbital -2 
        3:      Jump to or Continue along Orbital -1 
        4:      Jump to or Continue along Orbital +0 
        5:      Jump to or Continue along Orbital +1 
        6:      Jump to or Continue along Orbital +2 
        7:      Jump to or Continue along Orbital +3 

        8:      Slow Transfer to Own Base (should be masked if craft is unflagged)
        9:      Fast Transfer to Own Base (should be masked if craft is unflagged)

        10+2*k: Slow hit against opponent agent k
        11+2*k: Fast hit against opponent agent k

        NOTE: Even numbers that aren't transfers will always be slow hits, while odd numbers are fast hits

        :param np.ndarray action: the desired action to be taken by each mobile agent
        :returns: observations of the next step by calling get_obs(), rewards, terminated boolean, and truncated boolean (all three by calling reward()), info dictionary
        :rtype: dict
        """

        # Read in Opponent actions per mobile
        p0_action = actions['player0']
        p1_action = actions['player1']

        # if len(p0_action) != self._team_size or len(p1_action) != self._team_size:
        #     raise ValueError("Action size is incorrect for environment (ensure no actions are specified for bases, mobiles only).")

        # Start actions for player0
        for agent in range(0, self._team_size):
                act = int(p0_action[agent])
                # Agent is transferring; continue transfer (don't start anything)
                if act == 0:
                    continue

                # Start transfer to new orbital (act - 4 is the desired orbital)
                elif act in np.arange(1,8,1,dtype=int):
                    self._player0[agent+1].start_jump(act-4)

                # Agent returns to Player0's base via slow hit
                elif act == 8:
                    self._player0[agent+1].start_slow_hit(self._player0[0])

                # Agent returns to Player0's base via fast hit
                elif act == 9:
                    self._player0[agent+1].start_fast_hit(self._player0[0])

                # Agent slow hits an opponent's agent or base
                elif (act-10) % 2 == 0:
                    self._player0[agent+1].start_slow_hit(self._player1[int((act-10)/2)])

                # Agent fast hits an opponent's agent or base
                elif (act-11) % 2 == 0:
                    self._player0[agent+1].start_fast_hit(self._player1[int((act-11)/2)])

        # Start actions for player1
        for agent in range(self._team_size):
                act = int(p1_action[agent])

                # Agent is transferring; continue transfer (don't start anything)
                if act == 0:
                    continue

                # Start transfer to new orbital (act - 4 is the desired orbital)
                elif act in np.arange(1,8,1,dtype=int):
                    self._player1[agent+1].start_jump(act-4)

                # Agent returns to Player1's base via slow hit
                elif act == 8:
                    self._player1[agent+1].start_slow_hit(self._player1[0])

                # Agent returns to Player1's base via fast hit
                elif act == 9:
                    self._player1[agent+1].start_fast_hit(self._player1[0])

                # Agent slow hits an opponent's agent or base
                elif (act-10) % 2 == 0:
                    self._player1[agent+1].start_slow_hit(self._player0[int((act-10)/2)])

                # Agent fast hits an opponent's agent or base
                elif (act-11) % 2 == 0:
                    self._player1[agent+1].start_fast_hit(self._player0[int((act-11)/2)])

        # Update for every time step until next turn
        for _ in range(int(PTACT/PTCOL)):
            self._player0[0].propogate()
            self._player1[0].propogate()

            for agent in self._player0[1::]:
                agent.update()

            for agent in self._player1[1::]:
                agent.update()
                
            # Check for collisions
            if self._time % PTCOL == 0: 
                check_collisions(self)

            # Increment time  
            self._time += PTCOL

        # Get rewards for both agents
        p0_reward, p0_term, p0_trunc = self.reward("player0")
        p1_reward, p1_term, p1_trunc = self.reward("player1")

        # Get obs for both agents
        p0_obs = self.get_obs('player0')
        p1_obs = self.get_obs('player1')

        # Create masks for action space
        p0_mask, p1_mask = self.valid_action_masks()

        # Organize everything
        obs = {'player0':{'observation': p0_obs, 'action_mask':p0_mask}, 'player1':{'observation':p1_obs, 'action_mask':p1_mask}}
        reward = {'player0':p0_reward, 'player1':p1_reward}
        term = {'player0':p0_term, 'player1':p1_term}
        trunc = {'player0':p0_trunc, 'player1':p1_trunc}
 
        return obs, reward, term, trunc, {"player0":{}, "player1":{}}

    def render(self):
        """Produces a visual representation of the current game state"""
        # Sort agents based on status
        # Flagged
        flagged_0 = [unit for unit in self._player0[1::] if unit._flagged and not unit._transferring and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]
        flagged_1 = [unit for unit in self._player1[1::] if unit._flagged and not unit._transferring and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]

        # Dead or out of fuel
        dead_0 = [unit for unit in self._player0[1::] if not unit._alive or unit._fuel < MIN_FUEL_DICT[unit._orbital]]
        dead_1 = [unit for unit in self._player1[1::] if not unit._alive or unit._fuel < MIN_FUEL_DICT[unit._orbital]]

        # Transferring only
        transferring_0 = [unit for unit in self._player0 if unit._transferring == 1 and unit._flagged == 0 and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]
        transferring_1 = [unit for unit in self._player1 if unit._transferring == 1 and unit._flagged == 0 and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]

        # Flagged and transferring
        trans_flagged_0 = [unit for unit in self._player0[1::] if unit._flagged and unit._transferring and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]
        trans_flagged_1 = [unit for unit in self._player1[1::] if unit._flagged and unit._transferring and not unit._fuel < MIN_FUEL_DICT[unit._orbital]]

        # Normal
        normal_0 = [unit for unit in self._player0[1::] if unit._alive and unit._fuel > MIN_FUEL_DICT[unit._orbital] and not unit._flagged and not unit._transferring]
        normal_1 = [unit for unit in self._player1[1::] if unit._alive and unit._fuel > MIN_FUEL_DICT[unit._orbital] and not unit._flagged and not unit._transferring]

        # Collect polar angle and orbital information for each object

        R = lambda r: (norm(r) - R0) / 1000000

        # Bases
        base_0_r, base_0_t = self._player0[0]._orbital, self._player0[0]._angle
        base_1_r, base_1_t = self._player1[0]._orbital, self._player1[0]._angle

        # Dead agents
        dead_0_r = [R(x._posvector) for x in dead_0]
        dead_0_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in dead_0]
        dead_1_r = [R(x._posvector) for x in dead_1]
        dead_1_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in dead_1]

        # normal agents (not transferring)
        normal_0_r = [R(x._posvector) for x in normal_0]
        normal_0_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in normal_0]
        normal_1_r = [R(x._posvector) for x in normal_1]
        normal_1_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in normal_1]

        # transferring agents
        transferring_0_r = [R(x._posvector) for x in transferring_0]
        transferring_0_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2*np.pi) for x in transferring_0]
        transferring_1_r = [R(x._posvector) for x in transferring_1]
        transferring_1_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in transferring_1]

        # Flagged agents (not transferring)
        flagged_0_r = [R(x._posvector) for x in flagged_0]
        flagged_0_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in flagged_0]
        flagged_1_r = [R(x._posvector) for x in flagged_1]
        flagged_1_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in flagged_1]

        # flagged, transferring agents
        trans_flagged_0_r = [R(x._posvector) for x in trans_flagged_0]
        trans_flagged_0_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2*np.pi) for x in trans_flagged_0]
        trans_flagged_1_r = [R(x._posvector) for x in trans_flagged_1]
        trans_flagged_1_t = [(math.atan2(x._posvector[1], x._posvector[0])) % (2* np.pi) for x in trans_flagged_1]


        # Orbitals
        theta = np.linspace(0,2*np.pi,1000)
        R2 = lambda r: [r for _ in theta]
        rn3 = R2(-3)
        rn2 = R2(-2)
        rn1 = R2(-1)
        r0 = R2(0)
        r1 = R2(1)
        r2 = R2(2)
        r3 = R2(3)

        # Plot everything
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Plot orbitals
        ax.plot(theta, rn3, '--k', alpha=.3)
        ax.plot(theta, rn2, '--k', alpha=.5)
        ax.plot(theta, rn1, '--k', alpha=.7)
        ax.plot(theta, r0, '--k')
        ax.plot(theta, r1, '--k', alpha=.7)
        ax.plot(theta, r2, '--k', alpha=.5)
        ax.plot(theta, r3, '--k', alpha=.3)

        # Plot bases
        ax.plot(base_0_t, base_0_r, "sb", markersize=10)
        ax.plot(base_1_t, base_1_r, "sr", markersize=10)

        # Plot dead craft
        ax.plot(dead_0_t, dead_0_r, "xb")
        ax.plot(dead_1_t, dead_1_r, "xr")

        # Plot normal craft
        ax.plot(normal_0_t, normal_0_r, "^b")
        ax.plot(normal_1_t, normal_1_r, "^r")

        # Plot transferring craft
        ax.plot(transferring_0_t, transferring_0_r, "^b")
        ax.plot(transferring_1_t, transferring_1_r, "^r")

        # Plot Flagged Craft
        ax.plot(flagged_0_t, flagged_0_r, "*b")
        ax.plot(flagged_1_t, flagged_1_r, "*r")

        # Plot trans_flagged craft
        ax.plot(trans_flagged_0_t, trans_flagged_0_r, "*b")
        ax.plot(trans_flagged_1_t, trans_flagged_1_r, "*r")

        # Configure plot
        ax.set_ylim(-8,4)
        ax.set_xlim(0, 2*np.pi)
        ax.set_yticks(np.linspace(-3,3,7), ['-3','-2','-1', '+0','+1','+2','+3'], alpha=.5, fontsize=6)
        ax.set_xticks(np.linspace(0, 11*np.pi/6, 12), ['0', 'π/6', 'π/3', 'π/2', '2π/3', '5π/6', 'π', '7π/6', '4π/3', '3π/2', '5π/3', '11π/6'])
        ax.grid(False)
        ax.plot(0,-8,"og", markersize=20, alpha=.5, label="Earth")
        ax.spines['polar'].set_visible(False)

        # Save plot as array
        fig.canvas.draw()
        array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        return array

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

        # If agent has active mobiles but opponent doesn't
        if not all_dead_agent and all_dead_opponent:
            terminated = True
            reward += self._win_rew

        # If agent has no active mobiles but opponent does
        elif all_dead_agent and not all_dead_opponent:
            terminated = True
            reward -= self._win_rew

        # If neither the agent nor the opponent has any remaining active mobiles
        elif all_dead_agent and all_dead_opponent:
            truncated = True
        
        # Get rewards of events due to collisions
        for i in range(self._team_size):

            # If the agent's mobile acquires the flag
            if agent[i+1]._flagged and not agent[i+1]._got_flag: 
                reward += self._flag_rew
                self._player0[i+1]._got_flag = 1
            
            # If the opponent's mobile gets flag for the first time
            if opponent[i+1]._flagged and not opponent[i+1]._got_flag: 
                reward -= self._flag_rew
                self._player1[i+1]._got_flag = 1

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

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=-3, high=MAX_FUEL, shape=(2*(self._team_size+1), 8))

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.MultiDiscrete(np.array([self._num_actions]*self._team_size, dtype=np.int))
    
    def valid_action_masks(self):
        """Generates a mask for the action space for each player"""
        p0_mask = {i:np.array([1]*(self._num_actions), dtype=np.int8) for i in range(self._team_size)}
        p1_mask = {i:np.array([1]*(self._num_actions), dtype=np.int8) for i in range(self._team_size)}

        # Player0's mask
        for i in range(self._team_size):
            agent = self._player0[i+1]

            # If agent is out of fuel or dead, mask everything but action 0
            if not agent._alive or agent._fuel < MIN_FUEL_DICT[agent._orbital]:
                p0_mask[i][1:] = 0
                continue

            # If agent is transferring or carrying out a hit, mask everything but action 0
            if agent._transferring or agent._intercepting:
                p0_mask[i][1:] = 0
                continue

            else:
                # Check if has enough fuel to transfer to each orbital
                orbitals = [-3,-2,-1,0,1,2,3]
                for orbital in orbitals:
                    if agent._orbital == orbital or agent._fuel < SLOW_FUEL_COSTS[(agent._orbital, orbital)]:
                        p0_mask[i][orbital+4] = 0

                # Check if has enough fuel and is in range to initiate intercept against each object

                # Own base
                base = self._player0[0]
                if base._orbital == agent._orbital:
                    p0_mask[i][8:10] = 0

                # Slow intercept
                if not check_intercept(agent, base):
                    p0_mask[i][8] = 0

                # Fast Intercept
                if not check_intercept(agent, base, fast=True):
                    p0_mask[i][9] = 0

                # Enemy targets
                for k in range(self._team_size+1):
                    enemy = self._player1[k]

                    if agent._orbital == enemy._orbital:
                        p0_mask[i][10+2*k] = 0
                        p0_mask[i][11+2*k] = 0

                    # Slow intercept
                    if not check_intercept(agent, enemy):
                        p0_mask[i][10+2*k] = 0

                    # Fast Intercept
                    if not check_intercept(agent, enemy, fast=True):
                        p0_mask[i][11+2*k] = 0

        # Player1's mask
        for i in range(self._team_size):
            agent = self._player1[i+1]

            # If agent is out of fuel or dead, mask everything but action 0
            if not agent._alive or agent._fuel < MIN_FUEL_DICT[agent._orbital]:
                p1_mask[i][1:] = 0
                continue

            # If agent is transferring or carrying out a hit, mask everything but action 0
            if agent._transferring or agent._intercepting:
                p1_mask[i][1:] = 0
                continue

            else:
                # Check if has enough fuel to transfer to each orbital
                orbitals = [-3,-2,-1,0,1,2,3]
                for orbital in orbitals:
                    if agent._orbital == orbital or agent._fuel < SLOW_FUEL_COSTS[(agent._orbital, orbital)]:
                        p1_mask[i][orbital+4] = 0

                # Check if has enough fuel and is in range to initiate intercept against each object

                # Own base
                base = self._player1[0]
                if base._orbital == agent._orbital:
                    p1_mask[i][8:10] = 0

                # Slow intercept
                if not check_intercept(agent, base):
                    p1_mask[i][8] = 0

                # Fast Intercept
                if not check_intercept(agent, base, True):
                    p1_mask[i][9] = 0

                # Enemy targets
                for k in range(self._team_size+1):
                    enemy = self._player0[k]

                    if enemy._orbital == agent._orbital:
                        p1_mask[i][10+2*k] = 0
                        p1_mask[i][11+2*k] = 0

                    # Slow intercept
                    if not check_intercept(agent, enemy):
                        p1_mask[i][10+2*k] = 0

                    # Fast Intercept
                    if not check_intercept(agent, enemy, True):
                        p1_mask[i][11+2*k] = 0

        p0_mask = tuple(val for key,val in p0_mask.items())
        p1_mask = tuple(val for key,val in p1_mask.items())
        return p0_mask, p1_mask
    