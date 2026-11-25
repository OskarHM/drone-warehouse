<p align="center">
 <img width="350px" src="docs/img/rware.png" align="center" alt="Multi-Robot Warehouse (RWARE)" />
 <p align="center">A multi-agent reinforcement learning environment</p>
</p>

## Quick testing

From this repository, install the project with `python -m pip install -e .`, then run:

```sh
python human_play.py --env rware-tiny-2ag-v2 --display_info
```

Use the arrow keys to move and turn, **W/S** to ascend/descend, **P/L** to load/unload, and **Tab** to switch drones. Carry a green requested container to **G** at ground level; the delivery earns **+10** before action costs. Return it to an empty storage cell to unload. Press **D** to toggle terminal logs and **H** for all controls.

Each drone pays −0.05 per forward attempt, −0.15 per vertical attempt, and −0.01 per turn, wait or load/unload attempt. A blocked forward or vertical move costs an additional −0.20: for example, a blocked forward move returns −0.25. The default individual reward is logged as `Rew` for the selected drone;

For a short centralized-interface check, run `python smoke_test.py`.

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

> [!CAUTION]
> The RWARE environment was updated to support the new [Gymnasium](https://gymnasium.farama.org/) interface in replacement of the deprecated `gym=0.21` dependency (many thanks @LukasSchaefer). For backwards compatibility, please see [Gymnasium compatibility documentation](https://gymnasium.farama.org/content/gym_compatibility/) or use version v1.0.3 of the repository. The main changes to the interface are as follows:
> - `obss = env.reset()` --> `obss, info = env.reset()`
> - `obss, rewards, dones, info = env.step(actions)` --> `obss, rewards, done, truncated, info = env.step(actions)`
> - `done` is given as single boolean value instead of one `bool` value per agent
> - You can give the reset function a particular seed with `obss, info = env.reset(seed=42)` to initialise a particular episode.


<h1>Table of Contents</h1>

- [Environment Description](#environment-description)
- [Quick testing](#quick-testing)
  - [What does it look like?](#what-does-it-look-like)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Dynamics: Collisions](#dynamics-collisions)
  - [Rewards](#rewards)
- [Environment Parameters](#environment-parameters)
  - [Naming Scheme](#naming-scheme)
  - [Custom layout](#custom-layout)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Human Play](#human-play)
- [Please Cite](#please-cite)


# Environment Description

This project adapts the multi-robot warehouse (RWARE) environment for drones operating at ground and flight levels. The existing movable shelves represent containers: drones pick them up, deliver requested containers to a goal, and return them to empty storage locations.

The environment is configurable: it allows for different sizes (difficulty), number of agents, communication capabilities, and reward settings (cooperative/individual). Of course, the parameters used in each experiment must be clearly reported to allow for fair comparisons between algorithms.

## What does it look like?

The following illustration shows the original RWARE environment with four trained agents. The current renderer shows ground and flight levels side by side.

<p align="center">
 <img width="450px" src="docs/img/rware.gif" align="center" alt="Multi-Robot Warehouse (RWARE) illustration" />
</p>


## Action Space
Each drone has seven discrete actions:

`A = {Wait, Forward, Turn Left, Turn Right, Load/Unload, Ascend, Descend}`

Loading and unloading are only possible at ground level; unloading requires an empty storage location. An unloaded drone cannot ascend from beneath a container, and a drone cannot descend onto a stationary container. Drones can fly over containers at flight level.

## Observation Space
The default per-agent observation is local, covering a configurable square (3×3 by default) centred on the agent. It does not include altitude or fully represent flight-level occupancy:
- The location, the rotation and whether the agent is carrying a shelf.
- The location and rotation of other robots.
- Shelves and whether they are currently in the request queue.

For a centralized global observation that includes altitude and both drones, use `CentralizedWarehouse` from `rware.utils.wrappers`. See [required_changes_to_env.md](required_changes_to_env.md) for the state format and an example.

## Dynamics: Collisions
At ground level, unloaded drones can move beneath containers; loaded drones must avoid stationary containers. Drones may share the same horizontal position at different heights, but not the same `(x, y, z)` cell. A two-drone position swap is blocked.

Any collisions are resolved in a way that allows for maximum mobility. When two or more agents attempt to move to the same location, we prioritise the one that also blocks others. Otherwise, the selection is done arbitrarily. The visuals below demonstrate the resolution of various collisions.

 Example 1                 |   Example 2               | Example 3
:-------------------------:|:-------------------------:|:-------------------------:
![](docs/img/collision1.gif)  |  ![](docs/img/collision2.gif)  |  ![](docs/img/collision3.gif)

## Rewards
At any time, a set number of containers R is requested. A requested container delivered to a goal at ground level earns +10, and another container is sampled for the request queue. Action and blocked-move costs are listed in [Quick testing](#quick-testing). By default the reward is individual; global and two-stage reward modes are also available. In two-stage mode, +5 is awarded on delivery and +5 when the container is returned to storage, before action costs.

# Environment Parameters

The multi-robot warehouse task is parameterised by:

- The size of the warehouse which is preset to either tiny (10x11), small (10x20), medium (16x20), or large (16x29).
- The number of agents N.
- The number of requested shelves R. By default R=N, but easy and hard variations of the environment use R = 2N and R = N/2, respectively.

Note that R directly affects the difficulty of the environment. A small R, especially on a larger grid, dramatically affects the sparsity of the reward and thus exploration: randomly bringing the correct shelf becomes increasingly improbable.

## Naming Scheme

While RWARE allows fine tuning of multiple parameters when using the Warehouse class, it also registers multiple default environments with Gymnasium for simplicity.

Registered names look like `rware-tiny-2ag-v2`. They contain the map size, the number of agents and an optional `-easy` or `-hard` modifier, followed by the Gymnasium version `-v2`.

A few examples:
```python
import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2")
env = gym.make("rware-small-4ag-v2")
env = gym.make("rware-medium-6ag-hard-v2")
```


Of course, more settings are available, but have to be changed during environment creation. For example:
```python
import gymnasium as gym
import rware

env = gym.make("rware-tiny-2ag-v2", sensor_range=3, request_queue_size=6)
```

## Custom layout
You can design a custom warehouse layout with the following:
```python
import gymnasium as gym
import rware

layout = """
.......
...x...
..x.x..
.x...x.
..x.x..
...x...
.g...g.
"""
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
```
This will transform "X"s to shelves and "G"s to goal locations with a result like the one below:
<p align="center">
 <img width="300px" src="docs/img/rware-round.png" align="center" alt="Multi-Robot Warehouse (RWARE) illustration" />
</p>



A detailed explanation of all parameters can be found [here](https://github.com/semitable/robotic-warehouse/blob/4307b1fe3afa26de4ca4003fd04ab1319879832a/robotic_warehouse/warehouse.py#L132)

# Installation

Install this checkout in a Python virtual environment:
```sh
python -m pip install -e .
```

Installing `rware` from PyPI installs the upstream package, not necessarily this drone adaptation.

# Getting Started

RWARE was designed to be compatible with Open AI's Gym framework.

Creating the environment is done exactly as one would create a Gym environment:

```python
import gymnasium as gym
import rware
env = gym.make("rware-tiny-2ag-v2")
```

You can even bypass the `import` statement with Gym, and directly use:
```python
import gymnasium as gym
env = gym.make("rware:rware-tiny-2ag-v2")
```
The `rware:` in the beginning of the environment name tells Gym to import the respective package.

The number of agents, the observation space, and the action space are accessed using:
```python
env.unwrapped.n_agents  # 2
env.action_space  # Tuple(Discrete(7), Discrete(7))
env.observation_space  # Tuple(Box(XX,), Box(XX,))
```

The returned spaces are from the Gym library (`gym.spaces`) Each element of the tuple corresponds to an agent, meaning that `len(env.action_space) == env.n_agents` and `len(env.observation_space) == env.n_agents` are always true.

The reset and step functions again are identical to Gym:

```python
obs, info = env.reset()  # a tuple of observations and an info dictionary

actions = env.action_space.sample()  # the action space can be sampled
print(actions)  # (1, 0)
n_obs, rewards, terminated, truncated, info = env.step(actions)
print(terminated, truncated)
print(rewards)  # one reward per agent
```
Rewards are per agent; `terminated` and `truncated` are single booleans for the episode. A configured `target_deliveries` ends the task; `max_steps` and the optional inactivity limit truncate it.

Finally, the environment can be rendered for debugging purposes:
```python
env.render()
```
and should be closed before terminating:
```python
env.close()
```


# Human Play

We also provide a simple script that allows you to play the environment as a human. This is useful for debugging and understanding the environment dynamics. To play the environment, run the following command:
```sh
python human_play.py --env <env_name>
```
where `<env_name>` is the name of the environment you want to play. For example, to play the tiny environment with 2 agents, run:
```sh
python human_play.py --env rware-tiny-2ag-v2
```

Within the script, you can control a single agent at the time using the following keys:
- Up Arrow keys: move current agent forward
- Left/ Right Arrow keys: rotate current agent left/ right
- W / S: ascend / descend
- P/ L: pickup/ drop shelf
- SPACE: do nothing
- TAB: change the current agent (rotates through all agents)
- R: reset the environment and start a new episode
- H: show help
- D: display agent info (at every time step)
- ESC: exit


# Please Cite
If you use this environment, consider citing
1. A comperative evaluation of MARL algorithms that includes this environment
```
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```
2. A method that achieves state-of-the-art performance in the robotic warehouse task
```
@inproceedings{christianos2020shared,
 author = {Christianos, Filippos and Sch\"{a}fer, Lukas and Albrecht, Stefano},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {10707--10717},
 publisher = {Curran Associates, Inc.},
 title = {Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf},
 volume = {33},
 year = {2020}
}

```

