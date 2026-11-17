import pytest

from rware.warehouse import Action, Direction, RewardType, Warehouse


@pytest.fixture
def env():
    warehouse = Warehouse(3, 1, 1, 2, 0, 1, 1, None, None, RewardType.GLOBAL,
                          layout="....\n.XX.\n.gg.")
    warehouse.reset(seed=1)
    yield warehouse
    warehouse.close()


def place(env, first, second):
    for agent, (x, y, z) in zip(env.agents, (first, second)):
        agent.x, agent.y, agent.z = x, y, z
    env._recalc_grid()


def test_ascent_and_descent_restrictions(env):
    place(env, (1, 1, 0), (3, 0, 0))
    env.step([Action.ASCEND, Action.NOOP])
    assert env.agents[0].z == 0  # unloaded drone is under a container
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.ASCEND, Action.NOOP])
    assert env.agents[0].z == 1  # loaded drone carries it upward
    env.agents[0].dir = Direction.RIGHT
    env.step([Action.FORWARD, Action.NOOP])  # fly over another container
    assert env.agents[0].x == 2
    env.step([Action.DESCEND, Action.NOOP])
    assert env.agents[0].z == 1
    env.step([Action.FORWARD, Action.NOOP])
    env.step([Action.DESCEND, Action.NOOP])
    assert env.agents[0].z == 0


def test_same_xy_different_levels_and_vertical_swap(env):
    place(env, (0, 0, 0), (0, 0, 1))
    env.step([Action.ASCEND, Action.DESCEND])
    assert [a.z for a in env.agents] == [0, 1]
    env.step([Action.NOOP, Action.DESCEND])
    assert [a.z for a in env.agents] == [0, 1]


def test_elevated_load_does_not_deliver_or_block_ground(env):
    place(env, (1, 1, 0), (0, 0, 0))
    env.request_queue = [env.shelfs[0]]
    env.step([Action.TOGGLE_LOAD, Action.NOOP])
    env.step([Action.ASCEND, Action.NOOP])
    env.agents[0].dir = Direction.DOWN
    _, rewards, _, _, _ = env.step([Action.FORWARD, Action.NOOP])
    assert rewards == [0, 0]
    assert env.request_queue[0] is env.agents[0].carrying_shelf
    env.step([Action.DESCEND, Action.NOOP])
    # Delivery happens on arrival at the goal at ground level.
    assert env.agents[0].z == 0
    assert env.request_queue[0] is not env.agents[0].carrying_shelf
