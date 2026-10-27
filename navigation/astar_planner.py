import heapq
import numpy as np

class AStarPlanner:
    def __init__(self):
        pass

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, node, grid):
        neighbors = []
        x, y = node

        directions = [(1,0), (-1,0), (0,1), (0,-1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                if grid[nx][ny] == 0:
                    neighbors.append((nx, ny))

        return neighbors

    def find_path(self, grid, start, goal, risk_field=None):
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current, grid):

                if risk_field is not None:
                    risk_penalty = risk_field[neighbor[0]][neighbor[1]]
                else:
                    risk_penalty = 0

                tentative_g = g_score[current] + 1 + risk_penalty

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []