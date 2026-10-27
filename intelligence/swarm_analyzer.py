import numpy as np

class SwarmAnalyzer:
    def __init__(self):
        pass

    def compute_direction_instability(self, velocities):
        if len(velocities) == 0:
            return 0.0

        directions = []

        for vx, vy in velocities:
            angle = np.arctan2(vy, vx)
            directions.append(angle)

        directions = np.array(directions)

        # Measure spread of directions
        direction_variance = np.var(directions)

        # Normalize (higher variance = more chaotic movement)
        instability_score = min(direction_variance / 3.14, 1.0)

        return round(instability_score, 2)

    def compute_swarm_risk(self, density_score, instability_score):
        swarm_risk = 0.5 * density_score + 0.5 * instability_score
        return round(swarm_risk, 2)