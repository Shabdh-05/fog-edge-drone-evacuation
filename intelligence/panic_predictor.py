import numpy as np

class PanicPredictor:
    def __init__(self):
        self.previous_positions = {}

    def compute_velocity(self, person_id, current_position):
        if person_id not in self.previous_positions:
            self.previous_positions[person_id] = current_position
            return 0.0, 0.0

        prev_x, prev_y = self.previous_positions[person_id]
        curr_x, curr_y = current_position

        vx = curr_x - prev_x
        vy = curr_y - prev_y

        self.previous_positions[person_id] = current_position

        return vx, vy

    def compute_panic_index(self, velocity, density_score):
        speed = np.linalg.norm(velocity)

        # Normalize speed (assume 20 pixels/frame = high panic)
        speed_score = min(speed / 20.0, 1.0)

        # Combine with density
        panic_index = 0.6 * speed_score + 0.4 * density_score

        return round(panic_index, 2)