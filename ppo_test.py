from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from wrapped_env import CTFENVMA_sb3

# Initialize environment with 1 mobile per team, and reward of 1 for winning
env = CTFENVMA_sb3(1,1,0)

def get_mask(env):
    return env.action_mask()

env = ActionMasker(env, get_mask)

# Initialize maskable PPO model
model = MaskablePPO(MaskableActorCriticPolicy, env, seed=42, verbose=1)
eval_callback = MaskableEvalCallback(env, n_eval_episodes=10, eval_freq=10_000, log_path="./ppo_evaluations", use_masking=True) # Note: you will need to create a folder for the evaluations to be logged in
model.learn(total_timesteps=1e6, callback=eval_callback, progress_bar=True)
