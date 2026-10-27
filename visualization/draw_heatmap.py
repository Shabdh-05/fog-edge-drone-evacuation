import cv2
import numpy as np

class HeatmapVisualizer:
    def __init__(self):
        pass

    def apply_heatmap(self, frame, positions):
        heatmap = np.zeros_like(frame[:, :, 0])

        for x, y in positions:
            cv2.circle(heatmap, (int(x), int(y)), 40, 255, -1)

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

        return blended