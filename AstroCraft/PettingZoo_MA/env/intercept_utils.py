import numpy as np
import math
from poliastro.bodies import Earth
from itertools import product

from .constants import TRANSFERS, DHACTUAL, PDF, DHACTUAL, ORBITAL_ID, PTCOL

"""
Basic data used for both fast and slow intercept evaluations
"""

delta_angle_dict = dict()   # How far (in radians) the target drifts for each interval of length PTCOL
for orbital in ORBITAL_ID:
    orbital_period = 2 * math.pi * math.sqrt((DHACTUAL[orbital+3] ** 3)/Earth.k.value)
    ang_vel = 2 * math.pi / orbital_period
    delta_angle = ang_vel * PTCOL
    delta_angle_dict[orbital] = delta_angle

orbital_combos = [x for x in product(ORBITAL_ID, repeat=2) if x[0] != x[1]] # Tuples of form (a,b) representing an intercept from orbital a against a target on orbital b

angle_threshold_dict = dict()   # The maximum allowable error in starting angle for the agent and target to be within intercept distance of each other at the end of the intercept
for transfer in orbital_combos:
    orbital = transfer[1]
    offset = np.arctan(PDF/DHACTUAL[orbital+3])
    angle_threshold_dict[orbital] = offset

"""
Calculating the starting angle difference necessary for a slow (hohmann) intercept to occur
"""

slow = dict()
for transfer in orbital_combos:
    slow[transfer] = {'angle_diff': np.arccos(np.dot(TRANSFERS['slow'][transfer][0][0][0].value,TRANSFERS['slow'][transfer][0][0][-1].value) / (np.linalg.norm(TRANSFERS['slow'][transfer][0][0][0].value) * np.linalg.norm(TRANSFERS['slow'][transfer][0][0][-1].value))), 'tsteps': len(TRANSFERS['slow'][transfer][0][0])}

slow_target_drift_dict = dict()
for transfer, data in slow.items():
    target_orbital = transfer[1]
    target_delta_ang = delta_angle_dict[target_orbital]
    T = data['tsteps']

    # Target drifts T * delta_ang degrees
    drift = T * target_delta_ang
    slow_target_drift_dict[transfer] = drift

slow_angle_diff_dict = dict()
for transfer in slow.keys():
    target_drift = slow_target_drift_dict[transfer]
    agent_drift = slow[transfer]['angle_diff']
    slow_angle_diff_dict[transfer] = target_drift-agent_drift



"""
Calculating the starting angle difference necessary for a fast (lambert) intercept to occur
"""

fast = dict()
for transfer in orbital_combos:
    fast[transfer] = {'angle_diff': np.arccos(np.dot(TRANSFERS['fast'][transfer][0][0][0].value,TRANSFERS['fast'][transfer][0][0][-1].value) / (np.linalg.norm(TRANSFERS['fast'][transfer][0][0][0].value) * np.linalg.norm(TRANSFERS['fast'][transfer][0][0][-1].value))), 'tsteps': len(TRANSFERS['fast'][transfer][0][0])}

fast_target_drift_dict = dict()
for transfer, data in fast.items():
    target_orbital = transfer[1]
    target_delta_ang = delta_angle_dict[target_orbital]
    T = data['tsteps']

    # Target drifts T * delta_ang degrees
    drift = T * target_delta_ang
    fast_target_drift_dict[transfer] = drift

fast_angle_diff_dict = dict()
for transfer in fast.keys():
    target_drift = fast_target_drift_dict[transfer]
    agent_drift = fast[transfer]['angle_diff']
    fast_angle_diff_dict[transfer] = target_drift - agent_drift    

