from combat_env import CombatEnv

env = CombatEnv()
obs, info = env.reset()

print("Starting observation:", obs)

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: action={action}, reward={reward:.1f}, obs={obs}")

    if terminated or truncated:
        print("Episode ended! Resetting.")
        obs, info = env.reset()

env.close()