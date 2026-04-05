import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import warnings
from collections import deque

# Suppress Protobuf warnings for cleaner console
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# --- SECTION 1: THE BRAIN (Senses) ---
mp_hands = mp.solutions.hands
# Using model_complexity=0 and lower confidence for better hardware compatibility
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- SECTION 2: THE MEMORY (State) ---
shields_active = False          
gesture_timers = {'Left': 0, 'Right': 0} 
last_toggle_time = 0            
particles = []                  

# A 'deque' stores the last 6 hand positions to draw faded out shield ghosts
hand_trails = {'Right': deque(maxlen=6), 'Left': deque(maxlen=6)}

def is_v_gesture(hand_landmarks):
    """
    Checks if you are making a Peace/Victory sign.
    Using generous thresholds so it detects easily even if you are far from the camera.
    """
    wrist = hand_landmarks.landmark[0] 

    def get_dist(idx):
        tip = hand_landmarks.landmark[idx]
        return math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
    
    # 8=Index, 12=Middle, 16=Ring, 20=Pinky
    if get_dist(8) > 0.15 and get_dist(12) > 0.15: # Extended
        if get_dist(16) < 0.18 and get_dist(20) < 0.18: # Curled
            return True
    return False

def draw_detailed_mandala(img, center, base_radius, hand_angle, hand_label, alpha=1.0):
    """
    The Geometric Engine. Draws the circles, hexagons, and octagons.
    """
    if base_radius < 30: return 
    
    overlay = np.zeros_like(img)
    
    # Scale colors by 'alpha' and enforce integer types for Python 3.12
    amber = (0, int(140 * alpha), int(255 * alpha))
    gold = (0, int(210 * alpha), int(255 * alpha))
    core_w = (int(255 * alpha), int(255 * alpha), int(255 * alpha)) 

    spin_base = time.time() * 35 
    direction = 1 if hand_label == 'Right' else -1 
    
    # 1. DRAW STATIC RINGS
    cv2.circle(overlay, center, int(base_radius), amber, 2 if alpha < 1 else 4)
    cv2.circle(overlay, center, int(base_radius * 0.92), (0, int(100*alpha), int(210*alpha)), 2)

    # 2. DRAW OUTER OCTAGON
    pts_oct = []
    for i in range(8):
        angle = np.deg2rad((spin_base * 0.5 * direction) + hand_angle + (i * 45))
        px = int(center[0] + base_radius * 0.88 * math.cos(angle))
        py = int(center[1] + base_radius * 0.88 * math.sin(angle))
        pts_oct.append((px, py))
        
    for i in range(8): 
        cv2.line(overlay, pts_oct[i], pts_oct[(i+1)%8], amber, 2 if alpha < 1 else 3)

    # 3. DRAW INTERIOR HEXAGON
    pts_hex = []
    for i in range(6): 
        angle = np.deg2rad(-(spin_base * 1.5 * direction) + hand_angle + (i * 60))
        px = int(center[0] + base_radius * 0.75 * math.cos(angle))
        py = int(center[1] + base_radius * 0.75 * math.sin(angle))
        pts_hex.append((px, py))
        
    for i in range(6): 
        cv2.line(overlay, pts_hex[i], pts_hex[(i+1)%6], gold, 1 if alpha < 1 else 2)

    # 4. HOT CORE
    pulse = math.sin(time.time() * 6) * 4
    cv2.circle(overlay, center, max(1, int(base_radius * 0.1 + pulse)), core_w, -1)

    # 5. GLOW EFFECT
    blur_val = 65 if alpha > 0.8 else 31
    if blur_val % 2 == 0: blur_val += 1 # Ensure odd number for GaussianBlur
    glow = cv2.GaussianBlur(overlay, (blur_val, blur_val), 0)

    cv2.addWeighted(img, 1.0, glow, 0.7 * alpha, 0, img)
    cv2.addWeighted(img, 1.0, overlay, 1.0 * alpha, 0, img)

def update_sparks(frame, center, radius):
    """
    Creates and moves the little particles flying out from the shield.
    """
    global particles
    s_layer = np.zeros_like(frame)
    
    if radius > 30:
        for _ in range(2): 
            angle = random.uniform(0, 2*math.pi)
            px = int(center[0] + radius * math.cos(angle))
            py = int(center[1] + radius * math.sin(angle))
            vx = math.cos(angle)*4 + random.uniform(-1,1)
            vy = math.sin(angle)*4 + random.uniform(-1,1)
            particles.append([px, py, vx, vy, 180]) 

    new_p = []
    for p in particles:
        p[0], p[1], p[4] = p[0]+int(p[2]), p[1]+int(p[3]), p[4]-15 
        if p[4] > 0:
            # Boundary check to prevent crashes when sparks fly off-screen
            if 0 <= p[0] < frame.shape[1] and 0 <= p[1] < frame.shape[0]:
                cv2.circle(s_layer, (int(p[0]), int(p[1])), 1, (0, int(p[4]*0.6), int(p[4])), -1)
                new_p.append(p)
                
    particles = new_p
    cv2.addWeighted(frame, 1.0, s_layer, 1.0, 0, frame)

# --- SECTION 3: THE MAIN LOOP ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape    
    
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    active_labels = [] 
    
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lbl = results.multi_handedness[idx].classification[0].label 
            active_labels.append(lbl)
            
            wrist = hand_landmarks.landmark[0]
            mcp = hand_landmarks.landmark[9]
            thumb = hand_landmarks.landmark[4]
            pinky = hand_landmarks.landmark[20]
            
            cx, cy = int(((wrist.x + mcp.x)/2)*w), int(((wrist.y + mcp.y)/2)*h)

            # Check for the Toggle Gesture
            if is_v_gesture(hand_landmarks):
                if gesture_timers[lbl] == 0: 
                    gesture_timers[lbl] = time.time()
                elif (time.time() - gesture_timers[lbl]) > 0.5:
                    if (time.time() - last_toggle_time) > 1.2:
                        shields_active = not shields_active
                        last_toggle_time = time.time()
            else: 
                gesture_timers[lbl] = 0

            if shields_active:
                # Calculate Radius
                depth = math.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2)
                stretch = math.sqrt((thumb.x - pinky.x)**2 + (thumb.y - pinky.y)**2)
                radius = int((depth * 0.8 + stretch * 0.6) * w)
                
                # Calculate Angle
                angle = np.degrees(math.atan2(mcp.y - wrist.y, mcp.x - wrist.x))
                
                # Ghost trails memory
                hand_trails[lbl].append((cx, cy, radius, angle))
                
                history = list(hand_trails[lbl])
                # Draw the faded history
                for i, (tx, ty, tr, ta) in enumerate(history[:-1]):
                    draw_detailed_mandala(frame, (tx, ty), tr, ta, lbl, alpha=(i+1)/len(history)*0.4)
                
                # Draw main shield and particles
                draw_detailed_mandala(frame, (cx, cy), radius, angle, lbl, alpha=1.0)
                update_sparks(frame, (cx, cy), radius)

    # Clear memory for hands that leave the screen
    for l in ['Right', 'Left']: 
        if l not in active_labels: hand_trails[l].clear()
    # Define the label and the color (Green if ON, Red if OFF)
    status_text = f"SHIELD: {'ACTIVE' if shields_active else 'OFF'}"
    status_color = (0, 255, 0) if shields_active else (0, 0, 255)

    # Add text to the frame
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.imshow('Magic Shield Engine', frame)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()