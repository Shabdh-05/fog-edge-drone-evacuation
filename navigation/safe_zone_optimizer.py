class SafeZoneOptimizer:
    def __init__(self):
        pass

    def relocate_safe_zone(self, frame_width, frame_height, hazard_center, hazard_radius):
        hx, hy = hazard_center

        if hx < frame_width / 2:
            new_x = frame_width - 100
        else:
            new_x = 100

        if hy < frame_height / 2:
            new_y = frame_height - 100
        else:
            new_y = 100

        return (int(new_x), int(new_y))