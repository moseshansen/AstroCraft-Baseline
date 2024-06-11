import numpy as np
import math
from .moving_object import MovingObject
from .constants import PTCOL, TRANSFERS, SLOW_FUEL_COSTS, FAST_FUEL_COSTS, DHACTUAL
from .intercept_utils import delta_angle_dict
from .util import rotation_matrix
from poliastro.bodies import Earth

class MobileAgent(MovingObject):

    def __init__(self, posvector: np.ndarray, velvector: np.ndarray, team: int,
    num: int, alive: bool, fuel: float, flagged: bool, transferring: bool, orbital: int,
    angle: float, isbase: bool, verbose=False):
        super().__init__(posvector, velvector, team, angle, isbase, orbital, num)
        self._alive = int(alive)
        self._fuel = fuel
        self._flagged = int(flagged)
        self._transferring = int(transferring)
        self._intercepting = False
        self._angle = angle
        self._verbose = verbose
        self._returned_flag = False
        self._target = None
        self._got_flag = False
        self._team = team

    def calc_hohmann(self, target_orbital):
        """Calculates transfer points to carry out a hohmann (slow) intercept, if possible"""
        # Get transfer points
        r = TRANSFERS['slow'][(self._orbital, target_orbital)][0][0]
        v = TRANSFERS['slow'][(self._orbital, target_orbital)][0][1]
        
        d_angle = self._angle   # Transfer points assume starting angle of 0 rad, so store this angle to transform points later
        rot_mat = rotation_matrix(d_angle)

        # Calculate future positions, velocities, and fuel cost
        r_trajectory = [rot_mat @ ri for ri in r]
        v_trajectory = [rot_mat @ vi for vi in v]
        fuel_cost = SLOW_FUEL_COSTS[(self._orbital, target_orbital)]

        # Calculate angular trajectory
        t_trajectory = [(math.atan2(ri[1].value, ri[0].value)) % (2 * np.pi) for ri in r_trajectory]

        return r_trajectory, v_trajectory, t_trajectory, fuel_cost
    
    def calc_lambert(self, target_orbital):
        """Calculates transfer points to carry out a lambert (fast) intercept, if possible"""
        # Get transfer points
        r = TRANSFERS['fast'][(self._orbital, target_orbital)][0][0]
        v = TRANSFERS['fast'][(self._orbital, target_orbital)][0][1]

        d_angle = self._angle   # Transfer points assume starting angle of 0 rad, so store this angle to transform points later
        rot_mat = rotation_matrix(d_angle)

        # Calculate future positions, velocities, and fuel cost
        r_trajectory = [rot_mat @ ri for ri in r]
        v_trajectory = [rot_mat @ vi for vi in v]
        fuel_cost = FAST_FUEL_COSTS[(self._orbital, target_orbital)]

        # Calculate angular trajectory
        t_trajectory = [(math.atan2(ri[1].value, ri[0].value)) % (2 * np.pi) for ri in r_trajectory]

        return r_trajectory, v_trajectory, t_trajectory, fuel_cost
    
    def gen_transfer_points(self, target_orbital, transfer_type, intercept):
        """Generator for transfer points for any type of transfer or intercept"""

        # Get transfer points
        if transfer_type == 'hohmann':
            r_trajectory, v_trajectory, t_trajectory, fuel_cost = self.calc_hohmann(target_orbital)
        elif transfer_type == "lambert":
            r_trajectory, v_trajectory, t_trajectory, fuel_cost = self.calc_lambert(target_orbital)

        # Find out how many timesteps it will take for the transfer
        n_steps = len(r_trajectory)

        # Get angle at which the target will end up if it continues its current trajectory
        # We have to force the craft to end at this angle, since otherwise the intercept is slightly off
        if intercept:
            end_angle = self.get_end_angle(self._target, n_steps)
        
        # Yield new state until transfer is completed
        yield t_trajectory[0], self._orbital, r_trajectory[0].value, v_trajectory[0].value, self._fuel, False

        n_steps = len(r_trajectory)
        for i in range(1,n_steps-1):
            yield t_trajectory[i], self._orbital, r_trajectory[i].value, v_trajectory[i].value, self._fuel, False

        # For the last state, yield updated orbital and fuel level
        if intercept:
            r = DHACTUAL[target_orbital + 3] * np.array([np.cos(end_angle), np.sin(end_angle), 0])
            v = np.linalg.norm(v_trajectory[-1].value) * np.array([-np.cos(end_angle), np.sin(end_angle), 0])
            yield end_angle, target_orbital, r, v, self._fuel-fuel_cost, True

        else:
            yield t_trajectory[-1], target_orbital, r_trajectory[-1].value, v_trajectory[-1].value, self._fuel - fuel_cost, True

    def propagate(self):
        """Calculates transfer points for the agent's current orbit (i.e. agent is not 
        transferring or intercepting)"""

        # Get magnitude of change in angle
        delta_angle = delta_angle_dict[self._orbital]

        # Update Angle
        new_angle = (self._angle + delta_angle) % (2*np.pi)
        rot_mat = rotation_matrix(delta_angle)

        # Update Position and velocity
        r = DHACTUAL[self._orbital + 3]
        new_posvector = r * np.array([np.cos(new_angle), np.sin(new_angle), 0])
        new_velvector = np.linalg.norm(self._velvector) * np.array([-np.cos(new_angle), np.sin(new_angle), 0])

        return new_angle, self._orbital, new_posvector, new_velvector, self._fuel
        
    def start_jump(self, target_orbital):
        """Starts a hohmann transfer from the current orbital to the target orbital"""
        # Set transferring to True
        self._transferring = True

        # Get transfer points
        self.transfer_points = self.gen_transfer_points(target_orbital, 'hohmann', False)
        return

    def start_slow_hit(self, target):
        """Starts a slow hit (hohmann) from the current orbital to the opponent's orbital"""
        # Set intercepting to True
        self._intercepting = True
        self._transferring = True
        self._target = target
    
        # Get transfer points
        self.transfer_points = self.gen_transfer_points(target._orbital, 'hohmann', True)
        return
    
    def start_fast_hit(self, target):
        """Starts a fast hit (lambert) from the current orbital to the opponent's orbital"""
        # Set intecepting to True
        self._intercepting = True
        self._transferring = True
        self._target = target

        # Get transfer points
        self.transfer_points = self.gen_transfer_points(target._orbital, 'lambert', True)
        return

    def get_end_angle(self, target, n_steps):
        """Finds the angle at which the target will be after the intercept completes if it doesn't change trajectory"""
        delta_angle = delta_angle_dict[target._orbital]
        
        return (target._angle + delta_angle*n_steps) % (2*np.pi)
    
    def update(self):

        # If transferring or intercepting, get next set of values from generator.
        if self._transferring or self._intercepting:
            # Update until generator has finished
            self._angle, self._orbital, self._posvector, self._velvector, self._fuel, isDone = next(self.transfer_points)
            # if self._intercepting and self._target != None:
            #         print(f"ATTACKER:\tTeam {self._team} agent {self._num} pos: {self._posvector}\nTARGET\tTeam {self._target._team} agent {self._target._num} pos: {self._target._posvector}\nDISTANCE:\t{np.linalg.norm(self._posvector - self._target._posvector)}")

            if isDone:
                self._intercepting = False
                self._transferring = False 

        else:
            self._target = None
            self._angle, self._orbital, self._posvector, self._velvector, self.fuel = self.propagate()
