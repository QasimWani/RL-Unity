"""
    A wrapper that makes UnityEnvironment look like OpenAI Gym Environment
"""

from unityagents import UnityEnvironment


class UnityEnvWrapper:

    def __init__(self, path):
        self._env = UnityEnvironment(file_name=path)
        self.brain_name = self._env.brain_names[0]
        self.brain = self._env.brains[self.brain_name]

    def step(self, actions):
        """
            Args:
                actions (list): list of actions for each agent
            Returns:
                states (list): states observed by each agent
                rewards (list): reward recieved by each agent
                dones (list of bool): dones for each agent
        """
        env_info = self._env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return next_states, rewards, dones

    def reset(self):
        """
            Returns:
                States (list): states observed by each agent
        """
        env_info = self._env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations
