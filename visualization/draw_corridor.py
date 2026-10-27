import cv2

class CorridorVisualizer:
    def __init__(self):
        pass

    def draw_corridor(self, frame, corridor_points):
        for i in range(len(corridor_points) - 1):
            cv2.line(frame,
                     (int(corridor_points[i][0]), int(corridor_points[i][1])),
                     (int(corridor_points[i+1][0]), int(corridor_points[i+1][1])),
                     (0, 255, 0), 3)

        return frame