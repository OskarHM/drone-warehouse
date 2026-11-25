import pytest

from rware.utils.wrappers import CentralizedWarehouse
from rware.warehouse import Action, Direction, RewardType, Warehouse


@pytest.fixture
def env():
    warehouse = Warehouse(3, 1, 1, 2, 0, 1, 1, None, None, RewardType.INDIVIDUAL,
                          layout="....\n.XX.\n.gg.")
    warehouse.reset(seed=2)
    for agent, x in zip(warehouse.agents, (0, 3)):
        agent.x, agent.y, agent.z = x, 0, 0
    warehouse._recalc_grid()
    yield warehouse
    warehouse.close()


@pytest.mark.parametrize("action, direction, expected, blocked", [
    (Action.FORWARD, Direction.RIGHT, -0.05, False),
    (Action.FORWARD, Direction.LEFT, -0.25, True),  # boundary
    (Action.ASCEND, Direction.RIGHT, -0.15, False),
    (Action.DESCEND, Direction.RIGHT, -0.35, True),  # already on ground
    (Action.LEFT, Direction.RIGHT, -0.01, False),
    (Action.RIGHT, Direction.RIGHT, -0.01, False),
    (Action.NOOP, Direction.RIGHT, -0.01, False),
    (Action.TOGGLE_LOAD, Direction.RIGHT, -0.01, False),
])
def test_action_costs(env, action, direction, expected, blocked):
    env.agents[0].dir = direction
    _, rewards, _, _, info = env.step([action, Action.NOOP])
    assert rewards == pytest.approx([expected, -0.01])
    assert info["blocked_moves"] == (blocked, False)


def test_both_drones_are_penalized_for_blocked_swap(env):
    first, second = env.agents
    first.x, first.y, first.dir = 0, 0, Direction.RIGHT
    second.x, second.y, second.dir = 1, 0, Direction.LEFT
    env._recalc_grid()
    _, rewards, _, _, info = env.step([Action.FORWARD, Action.FORWARD])
    assert rewards == pytest.approx([-0.25, -0.25])
    assert info["blocked_moves"] == (True, True)


def test_global_team_reward_includes_both_action_costs():
    env = CentralizedWarehouse(Warehouse(3, 1, 1, 2, 0, 1, 1, None, None,
                                         RewardType.GLOBAL, layout="....\n.XX.\n.gg."))
    env.reset(seed=2)
    _, reward, _, _, info = env.step(Action.LEFT.value * 7 + Action.NOOP.value)
    assert reward == pytest.approx(-0.02)
    assert info["agent_rewards"] == pytest.approx((-0.01, -0.01))
    env.close()


def test_reward_ranges_include_two_blocked_vertical_moves(env):
    wrapped = CentralizedWarehouse(env)
    _, reward, _, _, info = wrapped.step(Action.DESCEND.value * 7 + Action.DESCEND.value)
    assert info["agent_rewards"] == pytest.approx((-0.35, -0.35))
    assert reward == pytest.approx(-0.70)
    assert env.reward_range[0] <= min(info["agent_rewards"])
    assert wrapped.reward_range[0] <= reward
