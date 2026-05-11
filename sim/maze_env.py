import numpy as np

maze = np.array([
    [0,0,0,0,0,0],
    [1,1,0,1,1,0],
    [0,0,0,0,1,0],
    [0,1,1,0,0,0],
    [0,0,0,1,1,0],
    [0,1,0,0,0,0]
])

ROWS, COLS = maze.shape

START_STATE = (0,0)
GOAL_STATE = (5,5)

ACTIONS = ["UP","DOWN","LEFT","RIGHT"]

def is_valid_state(row, col):

    if row < 0 or row >= ROWS:
        return False

    if col < 0 or col >= COLS:
        return False

    if maze[row][col] == 1:
        return False

    return True

def get_next_state(state, action):

    row, col = state

    if action == 0:
        new_row, new_col = row-1, col

    elif action == 1:
        new_row, new_col = row+1, col

    elif action == 2:
        new_row, new_col = row, col-1

    else:
        new_row, new_col = row, col+1

    if is_valid_state(new_row, new_col):
        return (new_row, new_col)

    return state

def get_reward(state):

    if state == GOAL_STATE:
        return 100

    return -1