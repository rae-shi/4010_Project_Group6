# Helpers 
A_LEFT, A_RIGHT, A_FORWARD = 0, 1, 2  # MiniGrid discrete actions

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])