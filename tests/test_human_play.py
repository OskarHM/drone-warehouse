import sys
from types import ModuleType, SimpleNamespace

import pytest

from human_play import InteractiveRWAREEnv
from rware.warehouse import Action, RewardType, Warehouse


@pytest.mark.parametrize("pressed, expected, starting_level", [
    ("W", Action.ASCEND, 0),
    ("S", Action.DESCEND, 1),
])
def test_altitude_keys_drive_environment(monkeypatch, pressed, expected, starting_level):
    # Key constants can be tested without opening a graphical window.
    keys = SimpleNamespace(LEFT=1, RIGHT=2, UP=3, W=4, S=5, P=6, L=7,
                           SPACE=8, TAB=9, R=10, H=11, D=12, ESCAPE=13)
    pyglet = ModuleType("pyglet")
    window = ModuleType("pyglet.window")
    window.key = keys
    pyglet.window = window
    monkeypatch.setitem(sys.modules, "pyglet", pyglet)
    monkeypatch.setitem(sys.modules, "pyglet.window", window)

    player = InteractiveRWAREEnv.__new__(InteractiveRWAREEnv)
    player.current_action = None
    player._key_press(getattr(keys, pressed), None)
    assert player.current_action == expected

    env = Warehouse(3, 1, 1, 1, 0, 1, 1, None, None, RewardType.GLOBAL,
                    layout="...\n.X.\n.gg")
    env.reset(seed=1)
    agent = env.agents[0]
    agent.x, agent.y, agent.z = 0, 0, starting_level
    env._recalc_grid()
    env.step([player.current_action.value])
    assert agent.z == 1 - starting_level
    env.close()
