import numpy as np
import pytest

from rware.utils.wrappers import CentralizedWarehouse
from rware.warehouse import Action, RewardType, Warehouse


def make_env(reward_type=RewardType.INDIVIDUAL, **kwargs):
    env = Warehouse(3, 1, 1, 2, 0, 1, 1, 5, 20, reward_type,
                    layout="....\n.XX.\n.gg.", **kwargs)
    return CentralizedWarehouse(env)


def test_observation_contains_full_state_and_tracks_altitude():
    env = make_env()
    observation, _ = env.reset(seed=4)
    base = env.unwrapped
    assert env.observation_space.contains(observation)
    assert env.action_space.n == 49

    # The global observation changes even when a distant shelf's request changes.
    base.request_queue = [base.shelfs[0]]
    first = env._global_observation()
    base.request_queue = [base.shelfs[1]]
    second = env._global_observation()
    assert not np.array_equal(first, second)

    base.agents[0].x, base.agents[0].y = 0, 0
    base.agents[0].z = 0
    base.agents[1].x, base.agents[1].y, base.agents[1].z = 3, 0, 0
    base._recalc_grid()
    before = env._global_observation()
    after, _, _, _, _ = env.step(Action.ASCEND.value * 7 + Action.NOOP.value)
    assert base.agents[0].z == 1
    agent_offset = 2 * np.prod(base.grid_size)
    assert before[agent_offset + 2] == 0
    assert after[agent_offset + 2] == 1
    assert after[-3:].tolist() == [1, 1, -1]
    assert env.observation_space.contains(after)
    env.close()


def test_observation_identifies_elevated_carried_container():
    env = make_env()
    env.reset(seed=1)
    base = env.unwrapped
    agent, other = base.agents
    carried, stationary = base.shelfs
    agent.x, agent.y, agent.z = stationary.x, stationary.y, 1
    carried.x, carried.y = agent.x, agent.y
    agent.carrying_shelf = carried
    other.x, other.y, other.z = 0, 0, 0
    base.request_queue = [carried]
    base._recalc_grid()
    state = env._global_observation()
    shelf_offset = 2 * np.prod(base.grid_size) + 6 * base.n_agents
    assert state[shelf_offset:shelf_offset + 4].tolist() == [agent.x, agent.y, 1, agent.id]
    assert state[shelf_offset + 4:shelf_offset + 8].tolist() == [stationary.x, stationary.y, 0, 0]
    env.close()


def test_joint_actions_address_each_agent_and_reject_invalid_values():
    env = make_env()
    env.reset(seed=1)
    agents = env.unwrapped.agents
    for agent, x in zip(agents, (0, 3)):
        agent.x, agent.y, agent.z = x, 0, 0
    env.unwrapped._recalc_grid()
    env.step(Action.NOOP.value * 7 + Action.ASCEND.value)
    assert [agent.z for agent in agents] == [0, 1]
    with pytest.raises(ValueError):
        env.step(49)
    env.close()


@pytest.mark.parametrize("reward_type", [RewardType.INDIVIDUAL, RewardType.GLOBAL])
def test_joint_reward_counts_delivery_once(reward_type):
    env = make_env(reward_type)
    env.reset(seed=3)
    base = env.unwrapped
    agent, other = base.agents
    shelf = base.shelfs[0]
    agent.x, agent.y, agent.z = 1, 1, 0
    shelf.x, shelf.y = 1, 1
    agent.carrying_shelf = shelf
    agent.dir = next(direction for direction in type(agent.dir) if direction.name == "DOWN")
    other.x, other.y, other.z = 3, 0, 0
    base.request_queue = [shelf]
    base._recalc_grid()
    _, reward, _, _, info = env.step(Action.FORWARD.value * 7 + Action.NOOP.value)
    assert reward == 1.0
    assert info["agent_rewards"] == ((1.0, 0.0) if reward_type == RewardType.INDIVIDUAL else (1.0, 1.0))
    env.close()


def test_delivery_target_terminates_and_progress_resets():
    env = make_env(target_deliveries=2)
    state, _ = env.reset(seed=3)
    base = env.unwrapped
    assert state[-1] == 0

    for expected in (1, 2):
        agent, other = base.agents
        shelf = base.request_queue[0]
        agent.x, agent.y, agent.z = 1, 1, 0
        shelf.x, shelf.y = 1, 1
        agent.carrying_shelf = shelf
        agent.dir = next(direction for direction in type(agent.dir) if direction.name == "DOWN")
        other.x, other.y, other.z = 3, 0, 0
        base._recalc_grid()
        state, reward, terminated, truncated, _ = env.step(Action.FORWARD.value * 7)
        assert reward == 1.0
        assert state[-1] == expected
        assert terminated == (expected == 2)
        assert not truncated
        # Clear the goal before arranging the next delivery.
        shelf.x, shelf.y = 0, 1
        agent.carrying_shelf = None

    state, _ = env.reset(seed=3)
    assert state[-1] == 0
    env.close()


def test_delivery_and_step_limit_can_happen_together():
    env = CentralizedWarehouse(Warehouse(3, 1, 1, 2, 0, 1, 1, None, 1,
                                       RewardType.INDIVIDUAL, layout="....\n.XX.\n.gg.",
                                       target_deliveries=1))
    env.reset(seed=3)
    base = env.unwrapped
    agent, other = base.agents
    shelf = base.request_queue[0]
    agent.x, agent.y, agent.z = 1, 1, 0
    shelf.x, shelf.y = 1, 1
    agent.carrying_shelf = shelf
    agent.dir = next(direction for direction in type(agent.dir) if direction.name == "DOWN")
    other.x, other.y, other.z = 3, 0, 0
    base._recalc_grid()
    _, _, terminated, truncated, _ = env.step(Action.FORWARD.value * 7)
    assert terminated and truncated
    env.close()


@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_delivery_target_must_be_positive_integer(target):
    with pytest.raises(ValueError):
        make_env(target_deliveries=target)
