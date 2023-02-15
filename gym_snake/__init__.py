from gymnasium.envs.registration import register

register(
     id="gym_snake/Snake-v0",
     entry_point="gym_snake.envs:SnakeEnv",
     max_episode_steps=1000,
)
