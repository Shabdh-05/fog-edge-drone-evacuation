from ml_modules.astar_path import AStarPathPlanner

# 0 = safe
# 1 = hazard/blocked

grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0]
]

planner = AStarPathPlanner(grid)

start = (0, 0)
goal = (4, 4)

path = planner.find_path(start, goal)

print("Safe Path:", path)