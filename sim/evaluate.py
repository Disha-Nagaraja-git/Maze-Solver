import pickle
import numpy as np

from maze_env import *

with open("../policies/policy_v1.pkl", "rb") as f:
    Q_table = pickle.load(f)

state = START_STATE

path = [state]

while state != GOAL_STATE:

    row,col = state

    action = np.argmax(Q_table[row,col])

    state = get_next_state(state, action)

    path.append(state)

print("Optimal Path:")
print(path)