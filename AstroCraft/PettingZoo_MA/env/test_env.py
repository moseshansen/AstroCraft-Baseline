from CaptureTheFlagMA import CTFENVMA

if __name__ == "__main__":
    env = CTFENVMA(3,10,0)
    states, infos = env.reset()
    actions = {agent: env.action_space(agent).sample(states[agent]['action_mask']) for agent in env.agents}
    print(actions)