# Import basic libraries
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#Import dependencies on Agent Libraries
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import TRPO, RecurrentPPO, QRDQN, ARS, MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from gymnasium.utils.env_checker import check_env as checkEnvGymnasium
from stable_baselines3.common.env_checker import check_env
# Import dependencies from our Env and misc
from ctfgymv3.CaptureTheFlag_v3 import CTFENV
from ctfgymv3.util import MAX_FUEL, PTACT, PCUBE, PTCOL, PTMAX, PRAD, R0, DH, DHACTUAL, check_collisions, MIN_FUEL_DICT, SLOW_FUEL_COSTS, FAST_FUEL_COSTS, check_intercept

# Masking function
def valid_action_masks(Env):
    """Generates a mask for the action space of each player by defaulting all values to 1.
        Set values to negative if the action is not allowed.
    """
    # Create masks of 1's
    p0_mask = {i:np.array([1]*(Env._num_actions), dtype=np.int8) for i in range(Env._team_size)}
    p1_mask = {i:np.array([1]*(Env._num_actions), dtype=np.int8) for i in range(Env._team_size)}

    # Player0's mask
    for i in range(Env._team_size):
        agent = Env._player0[i+1]

        # If agent is out of fuel or dead, mask everything but action 0
        if (not agent._alive) or (agent._fuel < MIN_FUEL_DICT[agent._orbital]):
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
            base = Env._player0[0]
            # if base._orbital == agent._orbital:
            #     p0_mask[i][8:10] = 0

            # Slow intercept
            if not check_intercept(agent, base):
                p0_mask[i][8] = 0

            # Fast Intercept
            if not check_intercept(agent, base, fast=True):
                p0_mask[i][9] = 0

            # Enemy targets
            for k in range(Env._team_size+1):
                enemy = Env._player1[k]

                # if agent._orbital == enemy._orbital:
                #     p0_mask[i][10+2*k] = 0
                #     p0_mask[i][11+2*k] = 0

                # Slow intercept
                if not check_intercept(agent, enemy):
                    p0_mask[i][10+2*k] = 0

                # Fast Intercept
                if not check_intercept(agent, enemy, fast=True):
                    p0_mask[i][11+2*k] = 0

    # Player1's mask
    for i in range(Env._team_size):
        agent = Env._player1[i+1]

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
            base = Env._player1[0]
            # if base._orbital == agent._orbital:
            #     p1_mask[i][8:10] = 0

            # Slow intercept
            if not check_intercept(agent, base):
                p1_mask[i][8] = 0

            # Fast Intercept
            if not check_intercept(agent, base, True):
                p1_mask[i][9] = 0

            # Enemy targets
            for k in range(Env._team_size+1):
                enemy = Env._player0[k]

                # if enemy._orbital == agent._orbital:
                #     p1_mask[i][10+2*k] = 0
                #     p1_mask[i][11+2*k] = 0

                # Slow intercept
                if not check_intercept(agent, enemy):
                    p1_mask[i][10+2*k] = 0

                # Fast Intercept
                if not check_intercept(agent, enemy, True):
                    p1_mask[i][11+2*k] = 0

    p0_mask = np.asarray(tuple(val for key,val in p0_mask.items()))
    #p1_mask = tuple(val for key,val in p1_mask.items())
    #masks = {'p0_mask': p0_mask, 'p1_mask':p1_mask}
    return p0_mask[0]

# Create the environment
env = CTFENV(team_size=1, opponent="RandomPlayer", win_rew=10, lose_rew=-10, flag_rew=1, lose_flag_rew=-1, logging=False, noisy=False, verbose=False)
env = ActionMasker(env, valid_action_masks)
# Check the environment against the Gymnasium checker
#checkEnvGymnasium(env)
# Check the environment against the Stable-Baselines3 checker
#check_env(env)
log_dir = "TestTensorBoard"
# Train the model 
model = MaskablePPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=3e6)
model.save("Final_Test") # Saves the final policy

# DRY RUN OF LEARNED POLICY
state, info = env.reset()
term = False
trunc = False
score = 0 

while not term or not trunc:
    action, _ = model.predict(env.get_obs(), deterministic=True)
    n_state, reward, term, trunc, info = env.step(action)
    score = score + reward
    state = n_state

