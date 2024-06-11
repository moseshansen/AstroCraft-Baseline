from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from wrapped_env import CTFENVMA_sb3

import numpy as np

# Initialize environment with 1 mobile per team, and reward of 1 for winning
env = CTFENVMA_sb3(1,1,0)
env = ActionMasker(env, "action_mask")

# Initialize maskable PPO model
model = MaskablePPO(MaskableActorCriticPolicy, env, seed=42, verbose=1)
model.learn(total_timesteps=1e9, callback=MaskableEvalCallback, progress_bar=True)
