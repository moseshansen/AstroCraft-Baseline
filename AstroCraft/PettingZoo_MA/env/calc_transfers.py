"""
This standalone code generates a table of orbital transfers. 
Two tranfer types are considered: Hohmann and Lambert. 
Each transfer method is calculated for seven Geosynchonous orbits. 
NOTE: poliastro version 0.17 will not work with this code. Use 0.16
NOTE: Need to comment out load_transfer_file() call in util for this script to work
"""
# Import libraries
import pickle
import json
import numpy as np
from astropy import units as u  # units
from util import PRAD, DHACTUAL, FAST_TIME, PTCOL
from itertools import permutations
from poliastro.bodies import Earth
from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
# Elliptical Kepler Equation solver based on a fifth-order
from poliastro.twobody.propagation import markley
from poliastro.core.maneuver import hohmann as hohmann_fast

# Designate Game orbits
orbitals = [-3, -2, -1, 0, 1, 2, 3]

# All possible orderings of transfers in pairs of 2 (initial orbit to final orbit), not repeated elements
combos = permutations(orbitals, 2)
combos = list(combos) # Transform to a readable list

# Create PoliAstro Manuever objects
hohmann = Maneuver.hohmann
lambert = Maneuver.lambert

# Create dictionaries for each type of transfer
slow_transfers = {}
fast_transfers = {}

print("calculating slow transfers...")

# For each transfer:
for combo in combos:
    # Create an initial circular orbit using the actual radius values of the first element of "combo" given in meters by DHACTUAL
    orbit = Orbit.circular(Earth, alt=(DHACTUAL[combo[0] + 3] - PRAD) * u.m)
    # Initiate a Hohmann transfer from the previously created orbit of radius given by combo[0] to radius given by combo[1]
    Htransfer = hohmann(orbit, DHACTUAL[combo[1]+3] * u.m)
    # Compute length of time for Hohmann transfer (NOTE that we cast to an int. As long as PTCOL is larger than 1sec, this is fine.)
    hTransferTime = int(Htransfer[1][0].value)
    # Generate a NumPy array of steps at which collisions will be checked throughout the transfer. Step size is PTCOL
    steps = np.arange(PTCOL, hTransferTime, PTCOL)
    # Add units of seconds to the values in the steps array 
    steps = steps* u.s
    # Add the Delta V required for the Hohmann transfer to the initial orbital velocity. Will be used to propagate transfer
    orbit.v += Htransfer[0][1]
    # Use Fifth-Order Markley solver to propagate the position and velocity vectors. 
    # Inputs are MU, initial position, final velcity, and steps array.
    r_list, v_list = markley(Earth.k, orbit.r.to(u.m), orbit.v.to(u.m / u.s), steps)
    # Store the lists of position and velocity points with the created Hohmann transfer in a Python dict indexed by the "combo" orbitals
    slow_transfers[(combo[0], combo[1])] = ((r_list, v_list), Htransfer)

print("calculating fast transfers...")

# For each transfer:
for combo in combos:
    # Create an initial circular orbit using the actual radius values of the first element of "combo" given in meters by DHACTUAL
    orbit = Orbit.circular(Earth, alt=(DHACTUAL[combo[0] + 3] - PRAD) * u.m)
    # Generate position and velocity vectors from the circular orbit. 
    rv = orbit.rv()
    # From the position vector (rv[0]), compute the angle between the x (rv[0][0]) and y (rv[0][1]) vectors. 
    # NOTE: that the angle will always be ZERO (due to the rv values generated) and in radians
    angle = np.arctan2(rv[0][1].value, rv[0][0].value) % (2*np.pi)
    # Convert Position from km to m and Velocity from km/s to m/s. Then, strip these units off of rv.
    rv = (rv[0].to(u.m).value, rv[1].to(u.m / u.s).value)
    # Compute a normal Hohmann transfer to generate two dv values (for burn maneuvers) and a transit time
    # Inputs are MU, position, velocity, and the TARGET altitude in meters (given by DHACTUAL)
    dv1, dv2, transitTime = hohmann_fast(Earth.k.value, rv, DHACTUAL[combo[1]+3])
    
    # Create the circular TARGET orbit using the actual radius values of the second element of "combo" given in meters by DHACTUAL
    # NOTE: The argument of latitude is pi/4 shifted for this orbit. TODO: Check why this is, with citation 
    new_orbit = Orbit.circular(Earth, alt=(DHACTUAL[combo[1]+3] - PRAD) * u.m, arglat=(angle + np.pi/4) * u.rad)
    # Initiate a Lambert Intercept from the previously created orbit of radius given by combo[0] to radius given by combo[1]
    # NOTE: This procedure required updating the poliastro maneuver.py file. TODO: Ask why 0.958...
    Ltransfer = lambert(orbit, new_orbit, tof=(transitTime * .958 * FAST_TIME) * u.s)  

    # Compute length of time for Lambert Intercept (NOTE that we cast to an int. As long as PTCOL is larger than 1sec, this is fine.)
    lTransferTime = int(Ltransfer[1][0].value)
    # Generate a NumPy array of steps at which collisions will be checked throughout the transfer. Step size is PTCOL
    lsteps = np.arange(PTCOL, lTransferTime, PTCOL)
    # Add units of seconds to the values in the steps array 
    lsteps = lsteps* u.s
    # Add the Delta V required for the Lambert Intercept to the initial orbital velocity. Will be used to propagate transfer
    orbit.v += Ltransfer[0][1]
    # Use Fifth-Order Markley solver to propagate the position and velocity vectors. 
    # Inputs are MU, initial position, final velcity, and lsteps array.
    r_list, v_list = markley(Earth.k, orbit.r.to(u.m), orbit.v.to(u.m / u.s), lsteps)
    # Store the lists of position and velocity points with the created Lambert Intercept in a Python dict indexed by the "combo" orbitals
    fast_transfers[(combo[0], combo[1])] = ((r_list, v_list), Ltransfer)

print("Saving transfers...")

'''
# Save all transfers to a Python dictionary
transfers = {"fast": fast_transfers, "slow": slow_transfers}
with open('TRANSFERS.json', 'w') as fp:
    json.dump(transfers, fp)
'''

transfers = {'fast': fast_transfers, 'slow': slow_transfers}
try:
    with open('./transfer_points/transfer_points_{}s.txt'.format(PTCOL), 'wb') as out_file:
       pickle.dump(transfers, out_file)
except:
    try:
        with open('ctfgymv3/transfer_points/transfer_points_{}s.txt'.format(PTCOL), 'wb') as out_file:
            pickle.dump(transfers, out_file)
    except:
        with open('../transfer_points/transfer_points_{}s.txt'.format(PTCOL), 'wb') as out_file:
            pickle.dump(transfers, out_file)

print("Saved!")
