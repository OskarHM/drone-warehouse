import math

import gymnasium as gym
import numpy as np

from rware.warehouse import Action


class CentralizedWarehouse(gym.Wrapper):
    """Expose the full warehouse state and one joint action for two agents.

    The float32 observation has a stable order:
    - row-major highway map, then row-major goal map;
    - for each agent by ID: x, y, z (0 in 2D), direction, carried shelf ID
      (0 means none), and the two-stage delivery flag;
    - for each shelf by ID: x, y, requesting flag, carrying agent ID (0 if
      stationary); carried shelves have the same x/y as their carrier;
    - elapsed steps, inactive steps, and deliveries completed (-1 when the
      corresponding limit or delivery target is off).

    Shelf IDs and agent IDs are one-based, as in RWARE. The fixed layout and
    episode limits are environment configuration. This wrapper requires two
    agents with discrete, message-free actions and supports the original 2D
    RWARE as well as the two-level version.
    """

    def __init__(self, env):
        super().__init__(env)
        warehouse = env.unwrapped
        spaces = env.action_space
        if (warehouse.n_agents != 2 or warehouse.msg_bits != 0
                or not isinstance(spaces, gym.spaces.Tuple)
                or len(spaces.spaces) != 2
                or not all(isinstance(space, gym.spaces.Discrete) and space.start == 0
                           for space in spaces.spaces)):
            raise ValueError("CentralizedWarehouse requires two agents with discrete, message-free actions")

        self._action_counts = tuple(int(space.n) for space in spaces.spaces)
        self.action_space = gym.spaces.Discrete(math.prod(self._action_counts))
        self._shelf_count = int(np.count_nonzero(warehouse.highways == 0))
        grid_cells = math.prod(warehouse.grid_size)
        observation_size = 2 * grid_cells + 6 * warehouse.n_agents + 4 * self._shelf_count + 3
        self.observation_space = gym.spaces.Box(
            low=-1, high=np.inf, shape=(observation_size,), dtype=np.float32
        )

    def _global_observation(self):
        warehouse = self.env.unwrapped
        goals = np.zeros(warehouse.grid_size, dtype=np.float32)
        for x, y in warehouse.goals:
            goals[y, x] = 1

        values = [*warehouse.highways.ravel(), *goals.ravel()]
        carriers = {}
        for agent in sorted(warehouse.agents, key=lambda agent: agent.id):
            shelf_id = agent.carrying_shelf.id if agent.carrying_shelf else 0
            if shelf_id:
                carriers[shelf_id] = agent.id
            values.extend((agent.x, agent.y, getattr(agent, "z", 0), agent.dir.value,
                           shelf_id, int(agent.has_delivered)))

        requests = {shelf.id for shelf in warehouse.request_queue}
        for shelf in sorted(warehouse.shelfs, key=lambda shelf: shelf.id):
            values.extend((shelf.x, shelf.y, int(shelf.id in requests),
                           carriers.get(shelf.id, 0)))

        values.extend((warehouse._cur_steps if warehouse.max_steps else -1,
                       warehouse._cur_inactive_steps if warehouse.max_inactivity_steps else -1,
                       warehouse.deliveries_completed if getattr(warehouse, "target_deliveries", None) is not None else -1))
        return np.asarray(values, dtype=np.float32)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._global_observation(), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid joint action: {action!r}")
        actions = divmod(int(action), self._action_counts[1])
        _, rewards, terminated, truncated, info = self.env.step(actions)
        # GLOBAL gives every agent the same team reward; summing would double it.
        if self.env.unwrapped.reward_type.name == "GLOBAL":
            reward = float(rewards[0])
        else:
            reward = float(sum(rewards))
        info = dict(info)
        info["agent_rewards"] = tuple(float(value) for value in rewards)
        return self._global_observation(), reward, terminated, truncated, info


class FlattenAgents(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        sa_action_space = [len(Action), *env.msg_bits * (2,)]
        if len(sa_action_space) == 1 and self.unwrapped.n_agents == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(
                self.unwrapped.n_agents * sa_action_space
            )
        self.action_space = sa_action_space

        self.observation_space = gym.spaces.Tuple(
            tuple(space for space in self.observation_space)
        )

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        return np.concatenate(
            [
                gym.spaces.flatten(s, o)
                for s, o in zip(self.observation_space, observation)
            ]
        ), info

    def step(self, action):
        try:
            action = np.split(action, self.unwrapped.n_agents)
        except (AttributeError, IndexError):
            action = [action]

        observation, reward, done, truncated, info = super().step(action)
        observation = np.concatenate(
            [
                gym.spaces.flatten(s, o)
                for s, o in zip(self.observation_space, observation)
            ]
        )
        reward = np.sum(reward)
        return observation, reward, done, truncated, info


class DictAgents(gym.Wrapper):
    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        digits = int(math.log10(self.unwrapped.n_agents)) + 1
        return {f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)}

    def step(self, action):
        digits = int(math.log10(self.unwrapped.n_agents)) + 1
        keys = [f"agent_{i:{digits}}" for i in range(self.unwrapped.n_agents)]
        assert keys == sorted(action.keys())

        # unwrap actions
        action = [action[key] for key in sorted(action.keys())]

        # step
        observation, reward, done, truncated, info = super().step(action)

        # wrap observations, rewards and dones
        observation = {
            f"agent_{i:{digits}}": obs_i for i, obs_i in enumerate(observation)
        }
        reward = {f"agent_{i:{digits}}": rew_i for i, rew_i in enumerate(reward)}
        done_dict = {
            f"agent_{i:{digits}}": done for i in range(self.unwrapped.n_agents)
        }
        truncated_dict = {
            f"agent_{i:{digits}}": truncated for i in range(self.unwrapped.n_agents)
        }

        return observation, reward, done_dict, truncated_dict, info


class FlattenSAObservation(gym.ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenSAObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = gym.spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return [
            gym.spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ]
