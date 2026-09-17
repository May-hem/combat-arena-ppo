import time
from combat_env import CombatEnv
from stable_baselines3 import PPO

model = PPO.load("combat_ppo")
env = CombatEnv(render_mode="human")

wins = 0
losses = 0
draws = 0

for episode in range(5):
    obs, info = env.reset()
    done = False
    steps = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
        time.sleep(0.1)

    if env.enemy_hp <= 0 and env.player_hp <= 0:
        result = "DRAW (mutual KO)"
        draws += 1
    elif env.enemy_hp <= 0:
        result = "LOSS"
        losses += 1
    elif env.player_hp <= 0:
        result = "WIN"
        wins += 1
    else:
        result = "DRAW (timeout)"
        draws += 1

    print(f"Episode {episode + 1}: {result} in {steps} steps (enemy_hp={env.enemy_hp:.0f}, player_hp={env.player_hp:.0f})")

env.close()
print(f"\nResults: {wins} wins, {losses} losses, {draws} draws out of 5")