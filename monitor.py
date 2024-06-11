from typing import Any, Dict, SupportsFloat, Tuple, List
from gymnasium import Env
from stable_baselines3.common.monitor import Monitor, ResultsWriter
import time

class CTFMonitor(Monitor):
    
    def __init__(self, env: Env, filename: str | None = None, allow_early_resets: bool = True, reset_keywords: Tuple[str, ...] = (), info_keywords: Tuple[str, ...] = (), override_existing: bool = True):
        super().__init__(env=env)
        self.t_start = time.time()
        self.results_writer = None
        if filename is not None:
            env_id = "CTFENVMA_sb3"
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=('c', 'f', 'p'),
                override_existing=override_existing,
            )

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards: List[float] = []
        self.needs_reset = True
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: Dict[str, Any] = {}
        self.episode_fuels: List[float] = []
        self.episode_proximities: List[float] = []
        self.episode_captures: List[bool] = []
        self.proximities: List[float] = []
        
    def reset(self, **kwargs) -> Tuple[Any | Dict[str, Any]]:
        self.proximities: List[float] = []
        return super().reset(**kwargs)
    
    def step(self, action: Any) -> Tuple[Any | SupportsFloat | bool | Dict[str, Any]]:
        
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(float(reward))
        self.proximities.append(abs(observation[1,-1].item()-observation[2,-1].item()))
        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6), "f": round(observation[1,3].item(), 6), "p": round(min(self.proximities), 6), "c": int(observation[1,4].item())}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            self.episode_fuels.append(observation[1,3].item())
            self.episode_proximities.append(min(self.proximities))
            self.episode_captures.append(bool(observation[1,4].item()))
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
            
        return observation, reward, terminated, truncated, info
    
    