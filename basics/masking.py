import cv2
import numpy as np
W, H = 800, 500
hidden = np.full((H, W, 3), (20, 10, 40), dtype=np.uint8)
text = "Unlocked OpenCV Mastery!!!"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
(tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
x = (W - tw) // 2
y = H // 2
cv2.putText(hidden, text, (x, y), font, scale,
            (255, 120, 200), thickness, cv2.LINE_AA)  # neon pink
sub = "pat pats yall doing so well"
scale2 = 0.7
(sw, sh), _ = cv2.getTextSize(sub, font, scale2, 1)
x2 = (W - sw) // 2
y2 = y + 40
cv2.putText(hidden, sub, (x2, y2), font, scale2,
            (200, 150, 255), 1, cv2.LINE_AA)  # lavender
cover = np.full((H, W, 3), (60, 170, 200), dtype=np.uint8)  # soft gold/teal
cv2.putText(cover, "SCRATCH HERE", (220, 270),
            font, 1.2, (30, 80, 100), 2, cv2.LINE_AA)

mask = np.zeros((H, W), dtype=np.uint8)

drawing = False
prev = None

def mouse(event, x, y, flags, param):
    global drawing, prev
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing, prev = True, (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(mask, prev, (x, y), 255, 50)
        cv2.circle(mask, (x, y), 25, 255, -1)
        prev = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, prev = False, None

cv2.namedWindow("Scratch Card")
cv2.setMouseCallback("Scratch Card", mouse)

while True:
    revealed = cv2.bitwise_and(hidden, hidden, mask=mask)
    covered  = cv2.bitwise_and(cover, cover, mask=cv2.bitwise_not(mask))
    frame = cv2.add(revealed, covered)

    cv2.putText(frame, "Drag | R reset | Q quit", (10, H-10),
                cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1)

    cv2.imshow("Scratch Card", frame)
    k = cv2.waitKey(16) & 0xFF
    if k == ord('q'): break
    if k == ord('r'): mask[:] = 0

cv2.destroyAllWindows()