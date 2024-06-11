"""
The util file defines routine functions for the environemnt.
"""

import math
import numpy as np

from .constants import FAST_FUEL_COSTS, SLOW_FUEL_COSTS, PSCORE, PDF, PTACT, PTCOL
from .intercept_utils import fast_angle_diff_dict, slow_angle_diff_dict, angle_threshold_dict, delta_angle_dict

def rotation_matrix(delta_angle:float):
    """
    Creates rotation matrix to rotate by an angle (delta_angle) about the z-axis
    :param float delta_angle: how much to rotate counterclockwise
    :returns: Rz rotation matrix for delta_angle
    :rtype: np.ndarray
    """

    r = np.stack((np.array([math.cos(delta_angle), -math.sin(delta_angle), 0]),
                  np.array([math.sin(delta_angle), math.cos(delta_angle), 0]),
                  np.array([0, 0, 1])))
    return r

def check_angle_diff(target_orbital, target_angle, agent_orbital, agent_angle, fast=False):
    """
    Checks if the angle difference between the agent and the target is within tolerance for the given intercept trajectory
    """
    # Find necessary difference (target-agent)
    if fast:
        necessary_diff = fast_angle_diff_dict[(agent_orbital,target_orbital)]
    else:
        necessary_diff = slow_angle_diff_dict[(agent_orbital,target_orbital)]

    thresh = angle_threshold_dict[target_orbital]

    # Check if true diff is within threshold of necessary diff
    true_diff = (target_angle-agent_angle)

    return (true_diff - necessary_diff) % (2*np.pi) <= thresh or (necessary_diff - true_diff) % (2*np.pi) <= thresh 

def check_intercept(agent, target, fast=False):
    """
    Verifies that the proposed intercept is possible by checking the following:
        -Agent has sufficient fuel to carry out the intercept
        -Angle between agents is correct for the proposed intercept
    Returns:
        Possible (bool): Whether or not the intercept is legal
    """

    # Check if target is transferring
    if target._transferring or not target._alive:
        return False

    # Check if agents are on same orbital
    if agent._orbital == target._orbital:
        return False

    # Check if agent has sufficient fuel
    agent_orbital = agent._orbital
    target_orbital = target._orbital

    if fast:
        req_fuel = FAST_FUEL_COSTS[(agent_orbital,target_orbital)]
    else:
        req_fuel = SLOW_FUEL_COSTS[(agent_orbital,target_orbital)]

    if agent._fuel < req_fuel:
        return False
    
    # Check if angle difference between the target and the agent is correct for the necessary intercept at any point during the next hour

    agent_delta_angle = delta_angle_dict[agent_orbital] # Change in angle during interval of PTCOL
    target_delta_angle = delta_angle_dict[target_orbital]   # Change in angle during interval of PTCOL

    temp_agent_angle = agent._angle
    temp_target_angle = target._angle

    for _ in range(PTACT // PTCOL):

        # Check if angles are right for the intercept to occur. Allow intercept if true for any timestep
        if check_angle_diff(target_orbital, temp_target_angle, agent_orbital, temp_agent_angle, fast):
            return True
        
        # If not, check angles at next timestep
        else:
            temp_agent_angle += agent_delta_angle
            temp_target_angle += target_delta_angle

    
    return False

def check_collisions(obj: object):
    """
    checks if two agents have collided, if they have, sets the agents alive and dead states as follows
    Pab = distance between agents
    v1 = agent1 velocity
    v2 = agent2 velocity
    if Pab * v1 > 0 and Pab * v2 > 0: agent 1 is eliminated
    if Pab * v1 < 0 and Pab * v2 > 0: both agents are eliminated
    if Pab * v1 < 0 and Pab * v2 < 0: agent 2 is eliminated
    :param obj: The environment object
    :type obj: object
    """
    # for every agent combination between teams
    for agent1 in np.concatenate((obj._player0, [obj._player1[0]])):
        for agent2 in np.concatenate((obj._player1, [obj._player0[0]])):

            if agent1._isbase and agent2._isbase:
                # Bases cannot hit one another
                continue

            # If either agent is dead, nothing happens.
            if not agent1._isbase:
                if not agent1._alive:
                    continue
            if not agent2._isbase:
                if not agent2._alive:
                    continue

            # represents the vector between agent 1 and 2
            rho_ab = agent2._posvector - agent1._posvector

            if np.linalg.norm(rho_ab) > PSCORE:
                # agents arent close enough together
                continue
            # can only hit or be hit by opponent MobileAgent, not own
            if agent1._team != agent2._team:
                # if within collision range and neither party is a base
                if np.linalg.norm(rho_ab) < PDF and not agent2._isbase and not agent1._isbase:
                    # If agent 1 is targetting agent 2, agent 2 is not transferring: 
                    if agent1._target == agent2:
                        # Kill agent 2 and reset agent 1 target
                        agent2._alive = False
                        agent1._target = None

                        if obj._verbose:
                            print('team {} agent {} killed team {} agent {}'.format(
                                agent1._team, agent1._num, agent2._team, agent2._num))

                    # If agent 2 is targetting agent 1, agent 1 is not transferring: 
                    if agent2._target == agent1:
                        # Kill agent 1 and reset agent 2 target
                        agent1._alive = False
                        agent2._target = None

                        if obj._verbose:
                            print('team {} agent {} killed team {} agent {}'.format(
                                agent2._team, agent2._num, agent1._team, agent1._num))

                # if within Scoring range and agent 2 is a base:
                if np.linalg.norm(rho_ab) < PSCORE and agent2._isbase and not agent1._flagged:
                    # Give agent 1 a flag and clear its target
                    agent1._flagged = True
                    agent1._target = None

                    if obj._verbose:
                        print('team {} agent {} is flagged'.format(
                            agent1._team, agent1._num))
                
                # if within Scoring range and agent 1 is a base:
                elif np.linalg.norm(rho_ab) < PSCORE and agent1._isbase and not agent2._flagged:
                    # Give agent 2 a flag and clear its target
                    agent2._flagged = True
                    agent2._target = None

                    if obj._verbose:
                        print('team {} agent {} is flagged'.format(
                            agent2._team, agent2._num))

            # Agent is targetting their own base
            else:
                # If Agent 1 is a mobile, with a flag, and Agent 2 is its base:
                if not agent1._isbase and agent1._flagged and agent2._isbase:
                    # Agent 1 returned a flag to their base
                    agent1._returned_flag = True
                    # Reset agent 1's target
                    agent1._target = None

                    if obj._verbose:
                        print('team {} agent {} intercepted home base'.format(
                            agent1._team, agent1._num))
                            
                # If Agent 2 is a mobile, with a flag, and Agent 1 is its base:
                if not agent2._isbase and agent2._flagged and agent1._isbase:
                    # Agent 2 returned a flag to their base
                    agent2._returned_flag = True
                    # Reset agent 2's target
                    agent2._target = None

                    if obj._verbose:
                        print('team {} agent {} intercepted home base'.format(
                            agent2._team, agent2._num))
