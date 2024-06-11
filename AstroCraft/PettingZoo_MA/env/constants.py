import pickle
import platform
import os
from poliastro.core.elements import coe_rotation_matrix, rv2coe, rv_pqw
from poliastro.bodies import Earth
from astropy import units as u
from numba import jit

"""
This constants file defines final constant values for the environment
"""

MAX_FUEL = 1000 #1000 orig
"""The maximum amount of fuel a MobileAgent has."""

MIN_FUEL_DICT = {-3: 39.958534550735756,
 -2: 38.49354796629723,
 -1: 37.11594323906411,
 0: 35.818593498774135,
 1: 34.59510220423863,
 2: 33.439713707558894,
 3: 33.44885154447093}
"""The minimum amount of fuel required for a MobileAgent to make a jump at each orbital 
(i.e. if the MobileAgent has less fuel than this level, they can take no actions and are 
effectively dead)."""

SLOW_FUEL_COSTS = {(-3, -2): 39.958534550735756,
 (-3, -1): 78.44832127686777,
 (-3, 0): 115.5435924791118,
 (-3, 1): 151.31831422137384,
 (-3, 2): 185.84116376422983,
 (-3, 3): 219.1760096052708,
 (-2, -3): 39.96945401541871,
 (-2, -1): 38.49354796629723,
 (-2, 0): 75.60628811033892,
 (-2, 1): 111.40612561489344,
 (-2, 2): 145.96109314342402,
 (-2, 3): 179.33446817965935,
 (-1, -3): 78.45904303547195,
 (-1, -2): 38.504067033322826,
 (-1, 0): 37.11594323906411,
 (-1, 1): 72.9318252930766,
 (-1, 2): 107.50988385483242,
 (-1, 3): 140.91281644238418,
 (0, -3): 115.55412588177705,
 (0, -2): 75.61662120929032,
 (0, -1): 37.12608579580319,
 (0, 1): 35.818593498774135,
 (0, 2): 70.41141813743207,
 (0, 3): 103.8356221637132,
 (1, -3): 151.3286679574567,
 (1, -2): 111.41628133761789,
 (1, -1): 72.94179266668834,
 (1, 0): 35.82838148384644,
 (1, 2): 34.59510220423863,
 (1, 3): 68.03292189172409,
 (2, -3): 185.85134592363144,
 (2, -2): 145.97107948826942,
 (2, -1): 107.51968395615042,
 (2, 0): 70.4210408685035,
 (2, 1): 34.604555807270124,
 (2, 3): 33.439713707558894,
 (3, -3): 219.1860277327883,
 (3, -2): 179.34429260534353,
 (3, -1): 140.92245664783832,
 (3, 0): 103.84508693847147,
 (3, 1): 68.04221939944546,
 (3, 2): 33.44885154447093}

FAST_FUEL_COSTS = {(-3, -2): 349.8525054910514,
 (-3, -1): 483.6735413663711,
 (-3, 0): 646.374117177228,
 (-3, 1): 814.9349609043862,
 (-3, 2): 982.377914771219,
 (-3, 3): 1146.3325157511863,
 (-2, -3): 349.85250549105194,
 (-2, -1): 342.9591842481555,
 (-2, 0): 470.0463241233488,
 (-2, 1): 625.9355081250934,
 (-2, 2): 788.1378250386215,
 (-2, 3): 949.647208004266,
 (-1, -3): 483.67354136637033,
 (-1, -2): 342.9591842481592,
 (-1, 0): 336.4563455430688,
 (-1, 1): 457.2504952787431,
 (-1, 2): 606.7158105244312,
 (-1, 3): 762.906920986854,
 (0, -3): 646.3741171772328,
 (0, -2): 470.04632412335036,
 (0, -1): 336.4563455430647,
 (0, 1): 330.3107376620553,
 (0, 2): 445.21744024592715,
 (0, 3): 588.6171913848012,
 (1, -3): 814.9349609043824,
 (1, -2): 625.9355081250933,
 (1, -1): 457.25049527873745,
 (1, 0): 330.3107376620553,
 (1, 2): 324.4926927774735,
 (1, 3): 433.88552838264803,
 (2, -3): 982.3779147712225,
 (2, -2): 788.1378250386215,
 (2, -1): 606.715810524427,
 (2, 0): 445.21744024592834,
 (2, 1): 324.49269277747317,
 (2, 3): 318.97567174162634,
 (3, -3): 1146.3325157511872,
 (3, -2): 949.6472080042662,
 (3, -1): 762.9069209868577,
 (3, 0): 588.6171913848018,
 (3, 1): 433.88552838265326,
 (3, 2): 318.97567174162236}
"""In the previous two dictionaries, the key-tuple represents the fuel cost of transferring from orbital i to orbital j"""

R0 = 42164000
"""Represents the orbital radius of the base orbital in meters."""

DH = [-3000, -2000, -1000, 0, 1000, 2000, 3000]
"""List of sizes of possible orbitals, with 0 representing R0 and the other values representing orbitals above and below R0, with their distances from R0 represented in meters."""

DHACTUAL = [R0-3000000, R0-2000000, R0-1000000,
            R0, R0+1000000, R0+2000000, R0+3000000]
"""List of sizes of possible orbits represented in their actual radius in meters."""

ORBITAL_ID = [-3, -2, -1, 0, 1, 2, 3]
"""List of the names of the orbitals, with 0 cooresponding to R0 and the others corresponding to values in DH."""

PDF = 120000 # Originally 100000, changed to 120000 to prevent missed intercepts due to precision issues when calculating intercepts.
"""Distance two mobile agents need to be in to collide."""

PTMAX = 28 * (23 * 3600 + 56 * 60)
"""The maximum game time in seconds, representing 28 sidereal days."""

PDT = 32
"""Time step in seconds for the game."""

PTACT = 3600
"""Unless prohibited by other rules, players can take an action every PTACT seconds."""

PSCORE = 700000 # originally 700000
"""Distance from a base in meters an agent must be to be flagged (if enemy base) or to win (if agent is flagged and base is home)."""

PTCOL = 36
"""Collisions and interactions are checked with a time step in seconds of PTCOL."""

PCUBE = 100000000
"""The game defined in 3D space of a cube with each side defined by PCUBE in meters."""

PRAD = 6378000
"""Represents the radius in meters of the earths surface, which exists at the center of PCUBE."""

FAST_TIME = .25
"""Fractions of a Hohmann transfer a fast transfer covers."""

DELAYED_PTCOL_STEPS = 0
"""Number of steps behind actual/most recent value"""


def load_transfer_file():
    """
    Loads in transfer data from transfer_points as saved by calc_transfers.py. util.py saves this in the TRANSFERS final variable.
    :returns: a dictionary of the transfer data
    :rtype: dict
    """
    if platform.system() == 'Linux' or platform.system() == 'Darwin':

        # path = np.__file__.strip('numpy/__init__.py')
        # dir_ = os.listdir("./transfer_points")
        # transfer_files = [i for i in dir_ if "_"+str(Ptcol)+'s' in i]
        # transfer_file = './transfer_points/' + transfer_files[0]

        # transfer_file = "/home/quickstunt/AstroCraft_Main/AstroCraft/ctfgymv3/transfer_points/transfer_points_"+str(PTCOL)+"s.txt"
        transfer_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_points/transfer_points_"+str(PTCOL)+"s.txt")

    elif platform.system() == 'Windows':
        # path = np.__file__.strip('numpy\\__init__.py')
        # dir_ = os.listdir(path+'\\transfer_points')
        # transfer_files = [i for i in dir_ if str(Ptcol) in i]
        # transfer_file = path + \
        #     '\\transfer_points\\' + transfer_files[0]

        transfer_file = "transfer_points\\transfer_points_"+str(PTCOL)+"s.txt"

    try:
        with open(transfer_file, "rb") as in_file:
            transfers = pickle.load(in_file)
    except:
        try:
            transfer_file = "env/transfer_points/transfer_points_"+str(PTCOL)+"s.txt"
            with open(transfer_file, "rb") as in_file:
                transfers = pickle.load(in_file)
        except:
            transfer_file = "../transfer_points/transfer_points_"+str(PTCOL)+"s.txt"
            with open(transfer_file, "rb") as in_file:
                transfers = pickle.load(in_file)

    return transfers

TRANSFERS = load_transfer_file()