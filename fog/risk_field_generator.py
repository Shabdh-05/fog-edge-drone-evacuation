import numpy as np

class RiskFieldGenerator:

    def __init__(self, w_density=0.4, w_panic=0.4, w_hazard=0.2):
        self.w_density = w_density
        self.w_panic = w_panic
        self.w_hazard = w_hazard

    def normalize(self, matrix):
        if np.max(matrix) == 0:
            return matrix
        return matrix / np.max(matrix)

    def generate_risk_field(self, density_map, panic_map, hazard_grid):
        
        density_norm = self.normalize(density_map)
        panic_norm = self.normalize(panic_map)
        hazard_norm = self.normalize(hazard_grid)

        risk_field = (
            self.w_density * density_norm +
            self.w_panic * panic_norm +
            self.w_hazard * hazard_norm
        )

        return risk_field