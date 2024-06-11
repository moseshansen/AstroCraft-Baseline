"""
This file contains the MovingObject class, which describes the base.
This is also a parent class for the MobileAgent class.
"""
import numpy as np
import math
from .constants import R0, PTCOL
from .util import rotation_matrix
from poliastro.bodies import Earth

class MovingObject:
    """
    :class: MovingObject
    
    The MovingObject class describes methods and attributes for objects that move. This includes the base as well as MobileAgents.
    """

    def __init__(self, posvector: np.ndarray, velvector: np.ndarray, team:int, angle:float, isbase: bool, 
        orbital: int, num: int):
        """
        Initializes the instance variables of the MovingObject instance.
        If values are None, it is set to the starting position/velocity of object of each team

        :ivar np.ndarray _posvector: the position vector of the MovingObject instance
        :ivar np.ndarray _velvector: the velocity vector of the MovingObject instance
        :ivar int _team: the MovingObject's team
        :ivar int _num: the MovingObject's number on its team
        :ivar int _orbital: describes the MovingObject instance's position with its orbital (-3 to 3)
        :ivar float _angle: describes the MovingObject instance's position with its angle from starting angle
        :ivar bool _isbase: if true then constructed object is a base, else it is a MobileAgent
        """

        self._team = team
        self._orbital = orbital
        self._angle = angle
        self._transferring = False
        self._intercepting = False
        self._isbase = int(isbase)
        self._orbital = orbital
        self._num = num

        if posvector is not None:
            self._posvector = posvector
        else:
            rot_mat = rotation_matrix(self._angle)
            self._posvector = rot_mat @ np.array([R0, 0, 0])
        if velvector is not None:
            self._velvector = velvector
        else:
            rot_mat = rotation_matrix(self._angle)
            self._velvector = rot_mat @ np.array([0, math.sqrt(Earth.k.value/R0), 0])
    
    def propogate(self):
        """
        The propagate method propagates the MovingObject instance on its circular trajectory.
        Overloaded for mobile_agents.
        This method is in-place.
        """
        if self._transferring == False:
            orbital_period = 2 * math.pi * \
            math.sqrt((np.linalg.norm(self._posvector)** 3)/Earth.k.value)

            ang_vel = 2 * math.pi / orbital_period

            delta_angle = ang_vel * PTCOL

            matrix = rotation_matrix(delta_angle)

            new_angle = self._angle + (delta_angle)
            if new_angle > 2 * np.pi:
                new_angle = new_angle % (2 * np.pi)
            self._angle = new_angle
            self._posvector = np.matmul(matrix, self._posvector)
            self._velvector = np.matmul(matrix, self._velvector)

        else: #the agent is transferring, so continue transfers
            if self._action == 'hoh':
                self.jump(self)
            elif self._action == 'lamb':
                self.fast_hit(self)

    def get_state(self):
        """
        The MovingObject and MobileAgent get_state method returns an array of the object's attribute data.
        This array is used in the environment get_obs() method.

        :returns: serialized MovingObject or MobileAgent data
        :rtype: np.array
        """
        # If the moving object is a base station, add in zero values for alive, fuel, and flagged.
        # If the moving object is a mobile agent, it's constructor will assign values for all parameters.
        if self._isbase:
            self._alive = 1
            self._fuel = 0
            self._flagged = 0
            self._num = 0
            self._transferring = 0
            self._intercepting = 0

        return np.array([self._team, self._num, self._alive, self._fuel, self._flagged, self._transferring, self._intercepting, self._orbital, self._angle]).astype(np.float32)