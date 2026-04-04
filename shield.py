import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
from collections import deque

# --- SECTION 1: THE BRAIN (Senses) ---
# MediaPipe's hand tracking model.
mp_hands = mp.solutions.hands
# detection confidence is to detect hand on screen and tracking is to track the 21 points on hand"
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

# --- SECTION 2: THE MEMORY (State) ---
shields_active = False          # shield is not turned on by default
gesture_timers = {'Left': 0, 'Right': 0} # how long did you keep V sign
last_toggle_time = 0            # just a timer to check because we dont want instant sensitive flickering
particles = []                  # all the flying sparks

# Colors in BGR (Blue, Green, Red). 
AMBER = (0, 140, 255)
GOLD = (0, 210, 255)
CORE_WHITE = (255, 255, 255)

# A 'deque' is a double ended queue (we set a max length). When it hits 6 items, it deletes the oldest one automatically.
# we store the last 6 hand positions to draw faded out shield so it appears like motion blur
hand_trails = {'Right': deque(maxlen=6), 'Left': deque(maxlen=6)}

def is_v_gesture(hand_landmarks):
    """
    Checks if you are making a Peace/Victory sign.
    Logic: Are the Index and Middle fingers long, and the Ring and Pinky fingers short?
    """
    wrist = hand_landmarks.landmark[0] # Landmark 0 is the wrist of your hand.
    # each landmark has .x .y .z attributes 
    # x and y range 0 to 1 ,   they tell the ratio with screen
    # so x=0 means to left and x=1 means right , y=0 means top and y=1 means bottom
    # z tells the depth ranging from -1 to 1 
    # here -ve will mean it is infront of wrist and +ve will mean it is behind wrist

    def get_dist(idx):
        # normal distance formula
        tip = hand_landmarks.landmark[idx]
        return math.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
    
    # 8=Index, 12=Middle, 16=Ring, 20=Pinky - the tips of fingers
    if get_dist(8) > 0.25 and get_dist(12) > 0.25: # Extended
        if get_dist(16) < 0.20 and get_dist(20) < 0.20: # Curled
            return True
    return False

def draw_detailed_mandala(img, center, base_radius, hand_angle, hand_label, alpha=1.0):
    """
    The Geometric Engine. Draws the circles, hexagons, and octagons.
    """
    if base_radius < 30: return # Stop if the hand is too far away.
    
    # Create a black 'overlay' frame to draw on. We later blend this with the camera frame.
    overlay = np.zeros_like(img)
    
    # alpha is the opacity
    # Scale colors by 'alpha' (0.0 to 1.0) so the trailing ghosts look transparent.
    amber = (0, int(140 * alpha), int(255 * alpha))
    gold = (0, int(210 * alpha), int(255 * alpha))
    core_w = (int(255 * alpha), int(255 * alpha), int(255 * alpha))   # for centre white pulse

    # spin_base increases every millisecond. This is why the shield rotates
    spin_base = time.time() * 35 
    direction = 1 if hand_label == 'Right' else -1 # Left hand spins the opposite way.
    
    # 1. DRAW STATIC RINGS (The simple circles)
    # cv2.circle(img, centre, radius, colour, thickness)
    cv2.circle(overlay, center, int(base_radius), amber, 2 if alpha < 1 else 4)
    cv2.circle(overlay, center, int(base_radius * 0.92), (0, int(100*alpha), int(210*alpha)), 2)

    # now we will draw the rotating shapes which are not circles
    # Math: Angle = (Time Rotation) + (Hand Tilt) + (degrees per point)
    # Trig: X = cos(angle), Y = sin(angle). This finds points on a circle.

    # 2. DRAW OUTER OCTAGON (8 points, spins clockwise)
    pts_oct = []
    for i in range(8):    # get 8 points for octagon 
        angle = np.deg2rad((spin_base * 0.5 * direction) + hand_angle + (i * 45))
        pts_oct.append((int(center[0] + base_radius * 0.88 * math.cos(angle)), 
                       int(center[1] + base_radius * 0.88 * math.sin(angle))))
    # Connect the 8 points with lines. The %8 connects point 7 back to point 0.
    for i in range(8): cv2.line(overlay, pts_oct[i], pts_oct[(i+1)%8], amber, 2 if alpha < 1 else 3)
    # cv2.line(img, start, end, colour, thickness)

    # 3. DRAW INTERIOR HEXAGON (6 points, spins counter-clockwise)
    pts_hex = []
    for i in range(6):    # get 6 points for hexagon
        angle = np.deg2rad(-(spin_base * 1.5 * direction) + hand_angle + (i * 60))
        pts_hex.append((int(center[0] + base_radius * 0.75 * math.cos(angle)), 
                       int(center[1] + base_radius * 0.75 * math.sin(angle))))
    for i in range(6): cv2.line(overlay, pts_hex[i], pts_hex[(i+1)%6], gold, 1 if alpha < 1 else 2)

    # 4. HOT CORE (The pulsing center dot)
    # basically doing sine of time will give us radius values which will show like smooth pulse
    pulse = math.sin(time.time() * 6) * 4
    cv2.circle(overlay, center, max(1, int(base_radius * 0.1 + pulse)), core_w, -1)

    # 5. GLOW EFFECT (The secret sauce)
    # We blur the overlay to create a kind of glow effect.
    blur_val = 65 if alpha > 0.8 else 31
    glow = cv2.GaussianBlur(overlay, (blur_val, blur_val), 0)
    # cv2.GaussianBlur(img, kernal dimensions, sigmaX=0)

    # Mix the original image + the glow + the sharp lines together.
    # overlay is the shield image we created which we will add on our original image(frame)
    # glow image is a soft blurred version of overlay
    # cv2.addWeighhted(original image, alpha of original, 
    #                   the other image we want to add, beta = other image alpha,
    #                   gamma, save result in this image)
    # img = alpha*img + beta*other + gamma
    cv2.addWeighted(img, 1.0, glow, 0.7 * alpha, 0, img)
    cv2.addWeighted(img, 1.0, overlay, 1.0 * alpha, 0, img)

def update_sparks(frame, center, radius):
    """
    Creates and moves the little particles flying out from the shield.
    """
    global particles   # access the global particles declared at top else it will create a new local variable
    s_layer = np.zeros_like(frame) # Temporary layer for sparks.
    if radius > 30:
        for _ in range(2): # Add 2 new sparks every single frame.
            angle = random.uniform(0, 2*math.pi)
            # starting position is on the outermost edge of the shield.
            px = int(center[0] + radius * math.cos(angle))
            py = int(center[1] + radius * math.sin(angle))
            # velocity: How fast the spark flies outwards. 
            # (we add this value on each frame so the spark keeps moving with constant velocity)
            vx, vy = math.cos(angle)*4 + random.uniform(-1,1), math.sin(angle)*4 + random.uniform(-1,1)
            particles.append([px, py, vx, vy, 180]) # 180 is the "Life" of the spark.
            # [ x coord, y coord,  X add value, y add value, time remaining for it to get removed]

    new_p = []
    for p in particles:
        # Move the particle based on its velocity.
        p[0], p[1], p[4] = p[0]+int(p[2]), p[1]+int(p[3]), p[4]-15 # -15 makes it age.
        if p[4] > 0: # if time remaining is still > 0 then only draw the spark
            cv2.circle(s_layer, (p[0], p[1]), 1, (0, int(p[4]*0.6), p[4]), -1)
            new_p.append(p)
    particles = new_p
    cv2.addWeighted(frame, 1.0, s_layer, 1.0, 0, frame)

# --- SECTION 3: THE MAIN LOOP  ---
cap = cv2.VideoCapture(0) # Open your webcam.
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1) # Flip the image so it feels like a mirror.
    h, w, _ = frame.shape    # output = height, width, no of channels
    # MediaPipe needs RGB images, but OpenCV uses BGR. We must convert!
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    active_labels = [] # List of hands seen in THIS frame.
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lbl = results.multi_handedness[idx].classification[0].label # Left or Right?
            active_labels.append(lbl)
            
            # Extract key landmarks. 
            # Wrist(0), Palm Knuckle(9), Thumb Tip(4), Pinky Tip(20).
            wrist, mcp, thumb, pinky = hand_landmarks.landmark[0], hand_landmarks.landmark[9], hand_landmarks.landmark[4], hand_landmarks.landmark[20]
            # Center the shield right in the middle of your palm.
            cx, cy = int(((wrist.x + mcp.x)/2)*w), int(((wrist.y + mcp.y)/2)*h)

            # Check for the Toggle Gesture.
            if is_v_gesture(hand_landmarks):
                if gesture_timers[lbl] == 0: gesture_timers[lbl] = time.time()
                # If held for 0.5 seconds, flip the switch.
                elif (time.time() - gesture_timers[lbl]) > 0.5:
                    if (time.time() - last_toggle_time) > 1.2:    # this is just to avoid wrong sensitive flickering
                        shields_active = not shields_active
                        last_toggle_time = time.time()
            else: gesture_timers[lbl] = 0

            if shields_active:
                # Calculate Radius: Scales with hand distance and how wide you spread your fingers.
                depth = math.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2)
                stretch = math.sqrt((thumb.x - pinky.x)**2 + (thumb.y - pinky.y)**2)
                radius = int((depth * 0.8 + stretch * 0.6) * w)
                # Calculate Angle: How much is your hand tilted?
                angle = np.degrees(math.atan2(mcp.y - wrist.y, mcp.x - wrist.x))
                
                # Add current position to trail memory.
                hand_trails[lbl].append((cx, cy, radius, angle))
                # Draw the older, "ghost" versions based on the last 6 hand positions
                history = list(hand_trails[lbl])
                for i, (tx, ty, tr, ta) in enumerate(history[:-1]):
                    draw_detailed_mandala(frame, (tx, ty), tr, ta, lbl, alpha=(i+1)/len(history)*0.4)
                
                # Draw the actual, sharp shield and sparks on top.
                draw_detailed_mandala(frame, (cx, cy), radius, angle, lbl, alpha=1.0)
                update_sparks(frame, (cx, cy), radius)

    # If a hand leaves the screen, clear its trail memory.
    for l in ['Right', 'Left']: 
        if l not in active_labels: hand_trails[l].clear()

    # Show the final result in a window.
    cv2.imshow('Shield', frame)
    # Press ESC to exit.
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()