from rware.warehouse import Action


def select_action(observation):
    return Action.RIGHT.value * len(Action) + Action.NOOP.value