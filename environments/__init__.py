from gym.envs.registration import register as gym_register


def register_env(id, **kwargs):
    try:
        gym_register(id=id, **kwargs)
    except Exception as e:
        print(f"Environment {id} already registered. Skipping.")


register_env(
    id="Robot-v1",
    entry_point="envs.robot:RobotMujocoEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)
