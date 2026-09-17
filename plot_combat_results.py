import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("combat_training_log.monitor.csv", skiprows=1)

plt.plot(data["l"])
plt.xlabel("Episode number")
plt.ylabel("Episode length (steps)")
plt.title("Combat Arena PPO Training Progress")
plt.savefig("combat_learning_curve.png")
plt.show()

print("Graph saved as combat_learning_curve.png")
