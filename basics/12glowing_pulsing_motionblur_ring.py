import cv2
import numpy as np
import math
import time
from collections import deque

trail = deque(maxlen=6)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    cx, cy = w // 2, h // 2

    # pulsing radius
    pulse = (math.sin(time.time() * 4) + 1)
    radius = int(20 + pulse * 40)

    trail.append((cx, cy, radius))

    # ghost trail layer
    ghost_layer = np.zeros_like(frame)

    for i, (tx, ty, tr) in enumerate(trail):
        alpha = (i + 1) / len(trail) * 0.5  # weaker for old frames

        cv2.circle(ghost_layer, (tx, ty), int(tr), (0, int(160 * alpha), int(255 * alpha)), 2)

    # main core layer
    core_layer = np.zeros_like(frame)
    cv2.circle(core_layer, (cx, cy), radius, (0, 215, 255), 2)

    #glow layer
    glow = cv2.GaussianBlur(core_layer, (31, 31), 0)

    # add all the layers
    frame = cv2.addWeighted(frame, 1.0, ghost_layer, 0.6, 0)
    frame = cv2.addWeighted(frame, 1.0, glow, 0.6, 0)
    frame = cv2.addWeighted(frame, 1.0, core_layer, 1.0, 0)

    cv2.imshow("Pulsing glowing ring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()