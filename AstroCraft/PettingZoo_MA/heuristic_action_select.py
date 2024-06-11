# Yield a random transfer
# Wait until base can be intercepted
# Intercept base
# Transfer randomly again
# Wait until home base can be intercepted
# Intercept home base

import numpy as np

def give_action(mask, t, has_flag):
    n_agents = len(mask)
    if t == 0:
        return np.random.choice([1,2,3,5,6,7], n_agents)
        #return np.array([7]*n_agents)

    else:
        action = np.zeros(n_agents) # Default to no action 

        # Find action for each agent
        for agent in range(n_agents):

            # If agent already has flag, use mask to check if on orbital 0
            if agent+1 in has_flag:
                if mask[agent][4] == 0: # Transfer to a different orbital
                    action[agent] = np.random.choice([1,2,3,5,6,7])
                    #action[agent] = 7
                else: # wait until intercept with home base is possible
                    p = np.random.rand()
                    if mask[agent][8] and p > .5:
                        print("A slow return to base is possible")
                        action[agent] = 8

                    elif mask[agent][9]:
                        print("A fast return to base is possible")
                        action[agent] = 9

            # If agent doesn't have flag, see if it can intercept base
            else:
                p = np.random.rand()
                if mask[agent][10] and p > .5:
                    print("A slow capture is possible")
                    action[agent] = 10

                elif mask[agent][11]:
                    print("A fast capture is possible")
                    action[agent] = 11

        return action            

