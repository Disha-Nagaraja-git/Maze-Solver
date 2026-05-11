import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import yaml
import json

from maze_env import *

with open("../configs/qlearning_v1.yaml", "r") as f:
    config = yaml.safe_load(f)

LEARNING_RATE = config["learning_rate"]
DISCOUNT_FACTOR = config["discount_factor"]
EPSILON = config["epsilon"]
EPISODES = config["episodes"]
MAX_STEPS = config["max_steps"]

Q_table = np.zeros((ROWS, COLS, 4))

episode_rewards = []
episode_steps = []

for episode in range(EPISODES):

    state = START_STATE
    total_reward = 0

    for step in range(MAX_STEPS):

        if random.uniform(0,1) < EPSILON:
            action = random.randint(0,3)

        else:
            row,col = state
            action = np.argmax(Q_table[row,col])

        next_state = get_next_state(state, action)

        reward = get_reward(next_state)

        row,col = state
        next_row,next_col = next_state

        old_q = Q_table[row,col,action]

        next_max = np.max(Q_table[next_row,next_col])

        new_q = old_q + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * next_max - old_q
        )

        Q_table[row,col,action] = new_q

        state = next_state

        total_reward += reward

        if state == GOAL_STATE:
            break

    episode_rewards.append(total_reward)
    episode_steps.append(step+1)

df = pd.DataFrame({
    "episode": range(1, EPISODES+1),
    "reward": episode_rewards,
    "steps": episode_steps
})

df.to_csv("../experiments/experiment_1.csv", index=False)

plt.plot(episode_rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Rewards Over Time")
plt.savefig("../plots/rewards.png")

plt.clf()

plt.plot(episode_steps)
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.title("Steps Per Episode")
plt.savefig("../plots/steps.png")

with open("../policies/policy_v1.pkl", "wb") as f:
    pickle.dump(Q_table, f)

log_data = {
    "episodes": EPISODES,
    "learning_rate": LEARNING_RATE,
    "discount_factor": DISCOUNT_FACTOR,
    "epsilon": EPSILON,
    "average_reward": float(np.mean(episode_rewards)),
    "average_steps": float(np.mean(episode_steps))
}

with open("../logs/log.json", "w") as f:
    json.dump(log_data, f, indent=4)

print("Training Complete")