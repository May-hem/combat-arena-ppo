from combat_env import CombatEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 1) Setting up the environment
env = CombatEnv()
env = Monitor(env, filename="combat_training_log")

# 2) Creating the agent
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01)
# 3) Training the agent
TIMESTEPS = 150000
model.learn(total_timesteps=TIMESTEPS)

# 4) Saving the trained model
model.save("combat_ppo")

env.close()
print(f"Training complete after {TIMESTEPS} timesteps. Model saved as combat_ppo.zip")