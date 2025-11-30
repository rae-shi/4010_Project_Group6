import numpy as np
import heapq
from helpers import A_LEFT, A_RIGHT, A_FORWARD

class HeuristicAgent:
    def __init__(self, env):
        self.width = env.unwrapped.width
        self.height = env.unwrapped.height
        
        # 0=Right (+1, 0), 1=Down (0, +1), 2=Left (-1, 0), 3=Up (0, -1)
        self.deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def plan_step(self, env_unwrapped):
        """
        1. check the environment state
        2. use heuristic function (BFS)
        3. return the action
        """ 
        grid = env_unwrapped.grid
        start_pos = tuple(env_unwrapped.agent_pos)
        start_dir = env_unwrapped.agent_dir

        # 1. find the goal position
        goal_pos = None
        for x in range(self.width):
            for y in range(self.height):
                c = grid.get(x, y)
                if c and c.type == 'goal':
                    goal_pos = (x, y)
                    break
        
        # If goal is somehow covered or missing, move forward
        if not goal_pos: return A_FORWARD

        # 2. BFS to find the shortest path
        # Queue: (cost, position, direction, action_history)
        queue = [(0, start_pos, start_dir, [])]
        visited = set()
        visited.add((start_pos, start_dir))

        while queue:
            cost, (cx, cy), c_dir, path = heapq.heappop(queue)

            # Check if we reached the goal 
            if (cx, cy) == goal_pos:
                if not path: return A_FORWARD # Already at goal
                return path[0] # Return the first step of the plan

            possible_actions = [A_LEFT, A_RIGHT, A_FORWARD]
            
            for action in possible_actions:
                nx, ny, ndir = cx, cy, c_dir
                
                # calculate the new state (position and direction)
                if action == A_LEFT:
                    ndir = (c_dir - 1) % 4 # left turn
                elif action == A_RIGHT:
                    ndir = (c_dir + 1) % 4 # right turn
                elif action == A_FORWARD:
                    dx, dy = self.deltas[ndir]
                    nx, ny = cx + dx, cy + dy # move coordinate

                # check the boundary
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell = grid.get(nx, ny)
                    
                    # check if the cell is walkable:
                    is_walkable = True
                    if cell:
                        # Agent cannot walk through Walls or Lava
                        if cell.type in ['wall', 'lava']: is_walkable = False
                    
                    # if the cell is walkable and the state is not visited, add to queue
                    if is_walkable and ((nx, ny), ndir) not in visited:
                        visited.add(((nx, ny), ndir))
                        new_path = path + [action]
                        # Cost is path length
                        heapq.heappush(queue, (cost + 1, (nx, ny), ndir, new_path))
        
        # if queue is empty and no path is found, default to forward
        return A_FORWARD