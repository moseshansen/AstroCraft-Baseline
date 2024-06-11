“DISTRIBUTION A: Approved for public release; distribution is unlimited. Public Affairs release approval # AFRL-2023-3267”

# Multi-Agent PettingZoo Environment with Masking
This is a PettingZoo implementation of the Astrocraft environment which allows for different strategies for both agents, including self-play, heuristics, etc.
This environment uses the same transfer mechanics as the Gymnasium environment, however there are the normal differences between Gymnasium and PettingZoo (explained in more detail below).

# Environment Usage
Here is a loose step-by-step on how to use the environment.

## Initialize the Environment
Initialize the environment normally, passing in the necessary arguments to the constructor. This environment is currently only capable of MvM games, however with a few small changes a MvN implementation should be simple enough to implement.

## Reset the Environment
Always call env.reset() before starting a new game. Reset will return two dictionaries: states and infos. 
States has the following structure: {'player0': {{'observation'}: np.ndarray, {'action_mask'}: tuple} 'player1': {{'observation'}: np.ndarray, {'action_mask'}: tuple}}.
The observation is a numpy array representing the current state of the game, and the action mask is a tuple of numpy arrays which mask the action space for each mobile. That is, states['player0']['action_mask'][2] may look something like [1,1,1,0,1,1,1,0,0,...], with the ones representing valid actions and the zeros representing invalid actions. 

Note also that the observation for both players is identical, however the order is swapped so that the agent's mobiles are first and the opponent's second.

## Generate an Action
This can be done several ways, but must always involve the mask. Attempting to pass in an invalid action will cause the environment to raise an exception and stop whatever it is doing, so be sure to use the mask to only pass in legal actions.
PettingZoo implements action spaces as a function rather than an attribute, so in order to sample randomly from the action space, you must specify which agent you're selecting an action for, and supply their mask. For example: env.action_space('player0').sample(mask). The way the mask is returned by the environment (i.e. a tuple of ndarrays) is the format required by the gymnasium.spaces module for masking of the multi-discrete space. 
To select an action via other means, you will need to figure out on your own how to apply the mask. This goes for scripted action selectors, neural networks, etc. 

## Step
Actions must be generated for both player0 and player1, though the means of generating the action can be different for the two (i.e. player0 could be neural net output while player1 is randomly sampled). Actions MUST then be passed to the step function as a dictionary of the following format: {'player0': <action>, 'player1': <action>}. This is standard for PettingZoo. 
The step function will execute the actions, check for collisions and intercepts, and calculate rewards and masks for both players. Due to the multi agent nature of the environment, observations, rewards, terminations, truncations, and infos are generated for both players and returned in dictionaries. 
The observation dictionary has the same structure as in the reset function. The other returned objects are also dictionaries with keys 'player0' and 'player1' mapping to the values corresponding to each player. Note that this means each agent will receive its own reward signal, and that it's possible for one agent to truncate before the other. 

# Visual_Debug.py
This is a handy script that simulates a game according to the configured action selectors for either player. The script renders the environment at each step and records the state. Once the game finishes, a GUI displays the renderings frame by frame alongside a stylized representation of the state during that frame. The user can step through the frames one at a time by using the left and right arrow keys, or 100 at a time with the up and down arrow keys. Closing the window (with either a click on the 'close' button in the top right corner or by pressing the escape key) will terminate the script.

# A Note on Compliance
Every effort has been made to ensure conformity with the standards and best practices for a PettingZoo environment, however due to issues with compatibility between the environment tester and environments that employ action masking (as of the writing of this README), do not expect the environment to pass such tests. 

