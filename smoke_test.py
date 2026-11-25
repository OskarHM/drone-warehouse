"""Run the current policy for four steps and print the results."""

import gymnasium as gym
import rware  # Registers the warehouse environments with Gymnasium.

from rware.utils.wrappers import CentralizedWarehouse
from train import select_action


def main():
    env = CentralizedWarehouse(
        gym.make("rware-tiny-2ag-v2", max_steps=4, disable_env_checker=True)
    )
    try:
        observation, _ = env.reset(seed=42)

        for step in range(1, 5):
            action = select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            heading = env.unwrapped.agents[0].dir.name
            print(
                f"step={step} action={action} heading={heading} "
                f"reward={reward} terminated={terminated} truncated={truncated}"
            )

            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
